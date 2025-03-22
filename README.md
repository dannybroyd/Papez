## An implementation of `Papez: Resource-Efficient Speech Separation with Auditory Working Memory` (ICASSP 2023)

> [Hyunseok Oh](http://www.aistudy.co.kr/ohs/), [Juheon Yi](https://juheonyi.github.io/), [Youngki Lee](http://youngkilee.blogspot.com/)<br>
> In ICASSP 2023. <br>

> Paper: https://ieeexplore.ieee.org/document/10095136<br>
> Slides: https://drive.google.com/file/d/1uksC183JlXdGwQ83rJgu-VFBNfLlE_r0/view?usp=sharing<br>
> Poster: https://drive.google.com/file/d/1h6wLwyAfA_A8xODHVKLI6zREkefI2h3c/view?usp=sharing<br>
> Video: https://drive.google.com/file/d/1hANUv-7_0S40A1jrfdRJ0yyR-FgNrwnv/view?usp=sharing<br>

> **Abstract:** *Transformer-based models recently reached state-of-the-art single-channel speech separation accuracy; However, their extreme computational load makes it difficult to deploy them in resource-constrained mobile or IoT devices. We thus present Papez, a lightweight and computation-efficient single-channel speech separation model. Papez is based on three key techniques. We first replace the inter-chunk Transformer with small-sized auditory working memory. Second, we adaptively prune the input tokens that do not need further processing. Finally, we reduce the number of parameters through the recurrent transformer. Our extensive evaluation shows that Papez achieves the best resource and accuracy tradeoffs with a large margin.*

### Usage

1. With Conda installed, run:

```
$ conda env create -f papez_env.yml
```

2. Activate the environment:

```
$ conda activate papez_env
```

3. Train a Papez model with (If you only have 1 gpu, ignore the flag):

```
$ python train.py --gpu $GPU_NUMBER
```
