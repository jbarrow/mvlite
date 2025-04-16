import modal

CACHE_DIR = "/hf-cache"
MINUTES = 60  # seconds

app = modal.App("colpali-encoder")

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        [
            "git+https://github.com/illuin-tech/colpali.git@782edcd50108d1842d154730ad3ce72476a2d17d",  # we pin the commit id
            "hf_transfer==0.1.8",
            "qwen-vl-utils==0.0.8",
            "torchvision==0.19.1",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": CACHE_DIR})
)

cache_volume = modal.Volume.from_name("colpali-cache", create_if_missing=True)


with modal_image.imports():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "aca78372505e6cb469c4fa6a35c60265b00ff5a4"


@app.function(
    image=modal_image, volumes={CACHE_DIR: cache_volume}, timeout=20 * MINUTES
)
def download_model():
    from huggingface_hug import snapshot_download

    result = snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    print(f"Model downloaded to {result}")


pdf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils")
    .pip_install("pdf2image==1.17.0", "pillow==10.4.0")
)


@app.function(image=pdf_image)
def convert_pdf_to_images(pdf_bytes):
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes, fmt="jpeg")
    return images


@app.cls(image=modal_image, gpu="L40S", volumes={CACHE_DIR: cache_volume})
class Model:
    @modal.enter()
    def setup(self):
        self.colqwen2_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.colqwen2_processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v0.1", use_fast=True
        )

    @modal.method()
    def encode(self, target: bytes | list):
        batch_size = 4

        if isinstance(target, bytes):
            images = convert_pdf_to_images.remote(target)
        else:
            images = target

        pdf_embeddings = []
        batches = [
            images[i : i + batch_size]
            for i in range(0, len(images), batch_size)
        ]
        for batch in batches:
            if isinstance(batch[0], str):
                encodings = self.colqwen2_processor.process_queries(batch).to(
                    self.colqwen2_model.device
                )
            else:
                encodings = self.colqwen2_processor.process_images(batch).to(
                    self.colqwen2_model.device
                )
            pdf_embeddings += list(
                self.colqwen2_model(**encodings).to("cpu").float().tolist()
            )

        return pdf_embeddings
