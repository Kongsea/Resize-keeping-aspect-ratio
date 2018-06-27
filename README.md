# Resize-to-fixed-size-keeping-aspect-ratio

Resize and pad images to a fixed size while keeping the original apspect ratio with tensor operations in TensorFlow

Please refer to [this Blog](http://www.yueye.org/2018/resize-an-image-in-tensorflow-while-keeping-the-aspect-ratio.html).

## 1.resize images to a fixed size

Use `tf.image.resize_images()`:

- `image = tf.image.resize_images(image, [new_height, new_width])`

But the original aspect ratio of images was changed.

## 2.pad images to a fixed size, or resize to fixed size keeping the original aspect ratio

Use `tf.image.pad_to_bounding_box`:

- `image = tf.image.pad_to_bounding_box(image, offset_height, offset_width, new_height, new_width)`

We need to calculate the offsets first and then pad images to the new sizes.
However, the padding values is contrined to 0.
We can modify the function to accept a parameter `constant_values` to set the padding values.
But we still cannot pad different values for different channels respectively.

## 3.pad specific values to resize images

- combine `tf.image.resize_images()` and `tf.image.pad_to_bounding_box`
- split 3 channels and pad every channel respectively
- concat 3 channels to restore images
