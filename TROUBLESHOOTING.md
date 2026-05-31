# Troubleshooting Guide — SRE/DevOps AI Assistant

Real issues encountered during project development, root causes, fixes, and lessons learned.

---

## Phase 1 — Data Collection

### Issue 1: Kubernetes sitemap returned sitemap index instead of URLs

**Symptom:** `k8s_raw.jsonl` was 0 bytes after scraper completed with no errors.

**Root cause:** `kubernetes.io/sitemap.xml` is a sitemap index (list of per-language sitemaps), not a direct URL list. The scraper parsed it but found no `/docs/` URLs because all actual URLs were nested one level deeper inside `kubernetes.io/en/sitemap.xml`.

**Fix:**
```python
# Step 1 — parse sitemap index to find English sitemap
index_soup = BeautifulSoup(resp.text, "xml")
english_sitemap = [loc.text for loc in index_soup.find_all("loc") if "/en/sitemap.xml" in loc.text][0]

# Step 2 — parse English sitemap to get actual /docs/ URLs
resp2 = requests.get(english_sitemap, headers=HEADERS)
urls  = [loc.text for loc in BeautifulSoup(resp2.text, "xml").find_all("loc") if "/docs/" in loc.text]
```

**Lesson:** Always validate scraper output early — 0 bytes with no error is a silent failure. Add a post-run assertion: `assert os.path.getsize(OUTPUT_FILE) > 0`.

---

### Issue 2: GitHub postmortem scraper returned 0 records

**Symptom:** `github_postmortems_raw.jsonl` had only 2 records — both from wrong repo.

**Root cause:** The regex `r'-\s+\[([^\]]+)\]\(([^)]+)\)'` expected list items starting with `-` but `danluu/post-mortems` uses plain paragraph format: `[Company](url). Description.` — no list prefix.

**Fix:**
```python
# Old — matched list items only
pattern = re.compile(r'-\s+\[([^\]]+)\]\(([^)]+)\)')

# New — matches plain paragraph links
pattern = re.compile(r'\[([^\]]+)\]\((https?://[^)]+)\)\.\s*(.+?)(?=\n\[|\n\n|\Z)', re.DOTALL)
```

**Lesson:** Always fetch and inspect 500 chars of the actual source before writing a parser. Never assume format from documentation — read the raw HTML/markdown.

---

### Issue 3: AWS Access Key ID scraped into training dataset

**Symptom:** GitHub Push Protection blocked push with error: `Amazon AWS Access Key ID found in data/raw/stackoverflow_raw.jsonl:716`.

**Root cause:** A Stack Overflow answer containing a real AWS key was scraped and included in the training dataset, then committed to git history.

**Fix:**
```bash
# Remove file from entire git history
git filter-repo --path data/raw/stackoverflow_raw.jsonl --invert-paths --force
git remote add origin https://github.com/...
git push origin main --force

# Add to .gitignore permanently
echo "data/raw/" >> .gitignore
echo "data/processed/" >> .gitignore
```

**Lesson:** Never commit scraped data to git. Training datasets frequently contain real credentials from public sources. Add `data/` to `.gitignore` before the first commit.

---

## Phase 2 — Dataset Preparation

### Issue 4: pgvector register_vector fails before CREATE EXTENSION

**Symptom:** `psycopg2.ProgrammingError: vector type not found in the database`

**Root cause:** `register_vector(conn)` was called in `get_connection()` before the `vector` extension was created in the database. The function queries the DB for the vector type OID — which doesn't exist yet.

**Fix:**
```python
# Wrong — register before extension exists
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)  # fails if extension not created yet
    return conn

# Correct — register AFTER CREATE EXTENSION
def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)  # now works
```

**Lesson:** pgvector type registration depends on database state. Always separate connection from type registration and call it after extension setup.

---

## Phase 3 — Fine-Tuning

### Issue 5: Unsloth compiled cache locked batch size

**Symptom:** `Batch size per device = 2` in Unsloth banner despite setting `per_device_train_batch_size=1` in TrainingArguments. OOM error on every training attempt.

**Root cause:** Unsloth compiles a custom SFTTrainer at instantiation time and caches it at `/kaggle/working/unsloth_compiled_cache/`. The cache stores the batch size from the first compilation. Subsequent changes to TrainingArguments don't invalidate the cache.

**Fix:**
```python
import shutil, os
cache_dir = '/kaggle/working/unsloth_compiled_cache'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared Unsloth cache")

# Also patch trainer args directly at runtime
trainer.args.per_device_train_batch_size = 1
trainer.args.train_batch_size = 1
```

**Lesson:** Always clear framework caches when changing hyperparameters. Add a `assert trainer.args.per_device_train_batch_size == 1` check before calling `.train()`.

---

### Issue 6: Colab/Kaggle compute limits interrupted training multiple times

**Symptom:** Training stopped at step 168 (Kaggle network disconnect), step 650 (Colab compute expired), step 700 (Colab compute expired again).

**Root cause:** Free tier GPU quotas — Colab resets daily, Kaggle sessions expire after 12 hours or on network disconnect. Checkpoints saved to ephemeral storage were lost on session reset.

**Fix:**
```python
# Push checkpoints to HuggingFace Hub on every save
class SaveToHubCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs["model"].push_to_hub(
                "username/model-checkpoints",
                commit_message=f"step-{state.global_step}",
                token=HF_TOKEN
            )
```

**Lesson:** For long training jobs on free-tier compute, always push checkpoints to persistent storage (HuggingFace Hub, Google Drive) every N steps. Never rely on ephemeral session storage.

---

### Issue 7: Merged model saved as bitsandbytes format instead of float16

**Symptom:** `convert_hf_to_gguf.py` failed with `NotImplementedError: Quant method is not yet supported: 'bitsandbytes'` despite using `save_method="merged_16bit"`.

**Root cause:** Unsloth's `save_pretrained_merged` saved the model with bitsandbytes quantization metadata in `config.json`. The actual weights contained `.absmax` tensors from the 4-bit quantization that weren't fully dequantized during merge.

**Fix — force dequantize in Colab:**
```python
# After merge_and_unload(), manually cast all parameters
for name, param in model.named_parameters():
    if param.dtype != torch.float16:
        param.data = param.data.to(torch.float16)

# Save locally first to force new weight files
model.save_pretrained('/content/sre-mistral-f16', safe_serialization=True)

# Remove quantization_config from config.json
with open('/content/sre-mistral-f16/config.json') as f:
    cfg = json.load(f)
cfg.pop('quantization_config', None)
with open('/content/sre-mistral-f16/config.json', 'w') as f:
    json.dump(cfg, f)

# Upload folder (forces new files)
HfApi().upload_folder(folder_path='/content/sre-mistral-f16', repo_id='username/model')
```

**Lesson:** `merge_and_unload()` doesn't always fully dequantize. Always verify weight dtypes after merge with `set(p.dtype for p in model.parameters())`. Use `upload_folder` not `push_to_hub` to force uploading new weight files.

---

### Issue 8: HuggingFace CDN cached stale config.json

**Symptom:** After updating `config.json` to remove `quantization_config`, the conversion script still read the old bitsandbytes config despite cache clearing.

**Root cause:** HuggingFace uses a CDN resolve-cache keyed on commit hashes. The `config.json` update via `HfApi().upload_file()` created a new commit hash but the conversion script's `--remote` flag was hitting the CDN which served the old commit.

**Fix:**
```bash
# Clear local HF cache completely
rm -rf ~/.cache/huggingface/hub/models--username--model-name

# Edit config.json directly in HuggingFace web UI
# huggingface.co/username/model → Files → config.json → Edit pencil icon
# Delete "quantization_config" block → Commit changes
```

**Lesson:** HuggingFace CDN caching can serve stale files. For config changes, edit directly in the web UI to ensure a clean commit. Always clear local HF cache before retrying conversion.

---

## Phase 4 — Local Serving

### Issue 9: Ollama pull failed for HuggingFace model

**Symptom:** `ollama pull Himanshu9001/sre-devops-mistral-7b` returned `Error: pull model manifest: file does not exist`

**Root cause:** `ollama pull` only works with models published in GGUF format on Ollama's registry or HuggingFace with proper GGUF files. The model was uploaded as safetensors, not GGUF.

**Fix — convert locally and create from GGUF:**
```bash
# Clone llama.cpp for conversion scripts
git clone https://github.com/ggerganov/llama.cpp.git
pip install gguf transformers

# Convert safetensors to GGUF
python convert_hf_to_gguf.py --outfile model-f16.gguf --outtype f16 --remote username/model

# Quantize to Q4_K_M
llama-quantize model-f16.gguf model-q4.gguf Q4_K_M

# Create Ollama model from local GGUF
cat > Modelfile << 'EOF'
FROM /absolute/path/to/model-q4.gguf
SYSTEM "You are an SRE expert..."