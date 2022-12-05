# Stable Tune, Fine tune your SD

Stable Tune is a tuner made to enable performant fine-tuning while providing better monitoring and training methodologies.
Based on Diffusers and Shivam's code, ST uses the latest memory-saving advancments and provides further QOL features such as:

Features

* **Fine Tuning Mindset** - ST is built to fine-tune, unlike Dreambooth, ST is meant to fine-tune a model, providing tools and settings to make most of your 3090/4090s.
* **Filename/Caption/Token based learning** - You can train using the induvidual file names as caption, use a caption txt file or a single token DB style, for finetuning file name and captions are best. 
* **Aspect Ratio Bucketing** - Using Everydream's Aspect Ratio bucketing you can use any aspect ratio or resolution for your training images, images will get shuffled into buckets and resized to your chosen resolution target!.
* **Remote monitoring using Telegram** - Want to keep tabs on your training? set a bot up in Telegram and receive samples and notifications as you train,  
* **Better Sampling controls** - To gauge how your model is doing sampling is important, to that effect ST gives you the option to add sample prompts as you see fit, set the number of images to produce per prompt, send a controlled seed prompt (to gauge how a seed changes) or even use random aspect ratios to see how buckets is changing your generations!.
* **Better Dataset Handling** - Use Dataset balancing to even out multiple concepts so they don't over-power each other, add class images to dataset to train them directly
* **Quality of life** - Many options to tune the experience to your liking, use save latent caching to avoid regenerating them at every run, use high batch-sizes to maximize training speed and performance and use epochs instead of steps to gauge progress better!.

![image](https://user-images.githubusercontent.com/87043616/204199775-e81e0b37-ed13-4795-99af-488314a58839.png)
![image](https://user-images.githubusercontent.com/87043616/204199622-e8c08886-b09e-4bc6-bd67-8031cd5b957f.png)
