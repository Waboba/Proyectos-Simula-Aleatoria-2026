# Difusión latente para la generación de imágenes de flores

Proyecto del curso MA4402 Simulación Estocástica.

Autores: **Felipe Cornejo** y **Rayén Lobos**.

Generamos imágenes de flores con difusión en el espacio latente de un autoencoder, estudiada como la simulación de una EDE reversa. El eje central es el muestreo determinista ↔ estocástico, controlado por el parámetro **η** (η=0 es DDIM/determinista, η=1 es DDPM/estocástico), junto con nuestro resultado de **invariancia de marginales** (todos los η llegan a la misma distribución). Un segundo parámetro, **w**, corrige la deriva del muestreo con un discriminador (*discriminator guidance*). Marco unificador: ambos son **perturbaciones de la deriva de la EDE reversa**.

## Estructura del repositorio

    pipeline_latente.ipynb                 # entrena el oráculo (U-Net), cachea latentes, muestreo con η
    discriminator_guidance_ordenado.ipynb  # discriminador, guía w, FID y heatmap η×w
    activaciones_pca.ipynb                 # PCA de activaciones, largo de arco (2.70×), segmentación de partes
    index.html / config.js                 # applet interactivo (η y w) — opcional
    extras/
        finetuning_adversarial.ipynb       # exploración (fine-tuning adversarial); NO se usa en la presentación

## Datos y checkpoints (externos)

Se descargan desde Drive:

- **Dataset de flores** (`flowers/`): https://drive.google.com/drive/folders/1tzfMvtlTQlYQoaerDlRwYahVnVc0GHGX?usp=sharing
- **Checkpoint del oráculo** (`latent_oracle.pt`): https://drive.google.com/file/d/1eGD4ra6QdspeqMjp39vwNMSsDp9ZuwdK/view?usp=drive_link
- **Checkpoint del discriminador** (`discriminador.pt`): https://drive.google.com/file/d/1PsRovikuI9PWJm7fDSGNRTkBHcax3l27/view?usp=drive_link

## Declaración de uso de IA

El **núcleo conceptual** del proyecto es nuestro: el resultado de invariancia de marginales y su demostración (vía Fokker-Planck), el marco de η y w como perturbaciones de la deriva de la EDE reversa, y la interpretación de los resultados.

Usamos IA como **apoyo** en:
- código de infraestructura (carga de pesos y datos, caché de latentes, decodificación por lotes, cálculo de FID, captura de activaciones con *hooks*, PCA y segmentación);
- limpieza y ordenamiento de notebooks;
- elección y uso de las librerías más complejas (`diffusers` para el VAE y los schedulers, `pytorch-fid` / `scipy` para el FID).

Dentro de los notebooks, los bloques hechos con apoyo de IA están marcados con el comentario **`# [IA]`**.

## Referencias

- Kim et al., *Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models*, ICML 2023.