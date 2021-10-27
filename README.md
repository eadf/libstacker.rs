# libstacker.rs
A port of the python code found [here: github.com/maitek/image_stacking](https://github.com/maitek/image_stacking) 

## Usage:
Download the test images:

```git clone https://github.com/maitek/image_stacking image_stacking_py```

Run the example:

```cargo run --release```

and then wait a minute..


## Todo

* Figure out what to do with .reshape() in keypoint_match()
* Figure out some opencv parameters
* Rayon (is opencv-rust MT-safe?)
* Make a proper lib
* Command line options in the example
