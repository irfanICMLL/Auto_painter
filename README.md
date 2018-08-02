# Auto_painter

It is the original implementation of the journal article:
Auto-painter: Cartoon image generation from sketch by using conditional Wasserstein generative adversarial networks
https://www.sciencedirect.com/science/article/pii/S0925231218306209?via%3Dihub

This project mean to make an end-to-end network for the sketch of cartoon to have color automatically.

More results can be seen here: https://irfanicmll.github.io/work-page/

Try our demo here: http://103.202.133.77:10086/

Since the lab's server has temporarily expired, the demo is now unavailable. You can see the demo video and train your own model.

New model has been updated!~ The performance is much better than in the orginal paper! See the demo video:https://youtu.be/g9rf-YFGgbg

Have a try~

The pre-train model can be download here: http://dsd.future-lab.cn\members\2016\Yifan\export_out3\model.rar

My homepage: http://dsd.future-lab.cn/members/2016/Yifan%20Liu.html

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

