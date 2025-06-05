import torch # type: ignore
from diffusers import StableDiffusionPipeline # type: ignore

# modelos Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
model_id2 = "CompVis/stable-diffusion-v1-4"
# modelos treinados
ModeloGuilhermeSemOculos = "./convercao/Guilherme_sem_oculos/GuilhermeSemOculosDiffusers"
ModeloGuilhermeComOculos = "./convercao/Guilherme_com_oculos/GuilhermeComOculosDiffusers"
ModeloGuilhermeMisto = "./convercao/Guilherme_Misto/GuilhermeMistoDiffusers"

device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(ModeloGuilhermeSemOculos, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Desativar filtro NSFW corretamente
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

prompt = "homem andando em uma floresta, quero uma foto realista."

guidance_scale = 12 # Mais precisão no prompt
num_steps = 70 # Melhor qualidade
# generator = torch.manual_seed(42) # Consistência
image = pipe(prompt, guidance_scale=guidance_scale,
            num_inference_steps=num_steps).images[0]  

image.save("HomemAndandoNaFloresta2.png")

#cd C:\Users\Guilherme\Documents\Projetos\Teste
#python Teste_de_imagen.py
#python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "./guilhermeModelo.ckpt" --original_config_file "v1-inference.yaml" --dump_path "./meuModeloDiffusers"

