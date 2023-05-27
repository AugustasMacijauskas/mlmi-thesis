from huggingface_hub import hf_hub_download


def main():
    hf_hub_download(repo_id="Dahoas/gptj-rm-static", filename="hf_ckpt.pt", cache_dir="../../.hf_cache/hub")

if __name__ == '__main__':
    main()