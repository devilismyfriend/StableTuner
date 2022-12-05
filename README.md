<p align="center">
    <br>
    <img src="https://i.imgur.com/DWIZXIh.png" width="800"/>
    <br>
<p>


# Stable Tuner, Fine-tune your SD
<a href='https://ko-fi.com/O4O5GU04F' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi2.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a> <a href='https://discord.gg/n8cKK7AAm4' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://cincydiscord.com/wp-content/uploads/2019/02/CINCYDISCORDJOIN.png' border='0' alt='Join the discord :)' /></a>


##Join the Discord for training and chill ;) 
Stable Tuner wants to be the easiest and most complete Stable Diffusion tuner :)

Features
* **For End Users** - ST was made to provide a solution that is convenient but powerful on windows, if you wanted to try finetuning, there's no better option, for Linux folks, a bash script will be added at a later date if there's enough interest.
* **Easy Installation** - ST makes installing convenient, using a bat file, ST will setup an environment ready for work and will install all the necessary components to get your training started fast!
* **Friendly GUI** - ST features a full GUI to configure training runs, import and export settings, view tool tips for options, test your new model in the playground, convert the model to CKPT and more!
* **Better Performance** - Using Diffusers, Xformers, CUDNN 1.8 and Bitsandbytes along with Latent caching allows for higher batch sizes and faster speeds, higher batch sizes = better quality model!.
* **Fine Tuning Mindset** - ST is built to fine-tune, unlike Dreambooth, ST is meant to fine-tune a model, providing tools and settings to make most of your 3090/4090s, Dreambooth is still an option.
* **Filename/Caption/Token based learning** - You can train using the individual file names as caption, use a caption txt file or a single token DB style, for finetuning file name and captions are best. 
* **Aspect Ratio Bucketing** - Using Everydream's Aspect Ratio bucketing you can use any aspect ratio or resolution for your training images, images will get shuffled into buckets and resized to your chosen resolution target!, supports up to 1024 resolution!.
* **Remote monitoring using Telegram** - Want to keep tabs on your training? set a bot up in Telegram and receive samples and notifications as you train,  
* **Better Sampling controls** - To gauge how your model is doing sampling is important, to that effect ST gives you the option to add sample prompts as you see fit, set the number of images to produce per prompt, send a controlled seed prompt (to gauge how a seed changes) or even use random aspect ratios to see how buckets are changing your generations!.
* **Better Dataset Handling** - Use Dataset balancing to even out multiple concepts so they don't over-power each other, add class images to dataset to train them directly, override per dataset if necessary.
* **Quality of life** - Many options to tune the experience to your liking, use save latent caching to avoid regenerating them at every run, use high batch-sizes to maximize training speed and performance, use epochs instead of steps to gauge progress better!.
* **Built for Diffusers** - ST uses HF's Diffusers library to allow the best and fastest implementations going forward, as of now, training 1.4,1.5,2 and 2-768 work great.

## Installation
Download and install Anaconda or miniconda and clone this repo, run the install_stabletuner.bat, when finished start the app with the StableTuner.cmd file.

##CUDNN 1.8
Due to the filesize I can't host the DLLs needed for CUDNN 1.8 on Github, I **strongly** advise you download them for a speed boost in sample generation (almost **50%** on 4090) you can download them from here: <a href="https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip">CUDNN 1.8</a>

To install simply unzip the directory and place in the same directory as StableTuner.cmd, run install_stabletuner.bat and you're good to go!

## Usage
Refer to the tool tips in the GUI for more information, if you have any questions feel free to ask in the <a href="https://discord.gg/n8cKK7AAm4">Discord</a>

## Kudos
* Shivam - For the original code and inspiration
* Diffusers - For the latest and greatest implementations
* Everydream - For the Aspect Ratio bucketing
* Sygil.dev - For the environment setup
* StabilityAI - For the latest and greatest models
* The whole SD community - For making this possible

## What's next?
* Linux support
* More models
* Advanced model mixing
* And more! :D
* Support me on Ko-Fi and come hang out in Discord to help me decide what's next :)