# Auto_painter
### News
We have released our dataset for public use.
The dataset can be downloaded through following links: 

Sketch-image pairs: https://cloudstor.aarnet.edu.au/plus/s/rMSBYCjEZJ70ab2

Sketch with control color blocks: https://cloudstor.aarnet.edu.au/plus/s/ixj8XS0rMmUqq0Z



Orginal README
---------------------------------------------------------------------------
It is the original implementation of the journal article:
Auto-painter: Cartoon image generation from sketch by using conditional Wasserstein generative adversarial networks 
https://www.sciencedirect.com/science/article/pii/S0925231218306209?via%3Dihub

This project mean to make an end-to-end network for the sketch of cartoon to have color automatically.

Try our demo here: http://103.202.133.77:10086/

Since the lab's server has temporarily expired, the demo is now unavailable. You can see the demo video and train your own model.
Or you can build your demo page based on our provided models following this project:
https://github.com/irfanICMLL/Auto_painter_demo

New model has been updated!~ The performance is much better than in the orginal paper! See the demo video:https://youtu.be/g9rf-YFGgbg

Have a try~

The pre-trained model can be downloaded from the following link: https://cloudstor.aarnet.edu.au/plus/s/LvyREKsiaH47Aa6

My homepage: https://irfanicmll.github.io/

Welcome to contact me~


### Dependencies

python3.5

tensorflow1.4

Vgg model from:https://github.com/machrisaa/tensorflow-vgg(optional, if you use the loss_f)

### Data
Color images: Collected on the Internet

Sketch: Generated from the preprocessing/gen_sketch/sketch.py


### Quick start

Put you orginal data in the folder preprocessing/gen_sketch/pic_org 

Run the sketch.py and you will get the training set in the preprocessing/gen_sketch/pic_sketch folder

Download the pre-train weight of Vgg16, and put the model and the pretrian weight uder the folder of training&test/my_vgg

Run the training command as:

python auto-painter.py --mode train --input_dir $TRAINING_SET --output_dir $OUTPUT --checkpoint None

Run the testing command as:

python auto-painter.py --mode test --input_dir $TESTING_SET --output_dir $OUTPUT_TEST --checkpoint $OUTPUT

