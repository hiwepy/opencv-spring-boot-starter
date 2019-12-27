/*
 * Copyright (c) 2018, hiwepy (https://github.com/hiwepy).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.bytedeco.opencv.spring.boot.nd4j;

/**
 * @author ： <a href="https://github.com/hiwepy">wandl</a>
 */
public enum ImageLoader {
	
	/**
	 * CifarLoader is loader specific for the Cifar10 dataset
	 *
	 * Reference: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
	 *
	 * There is a special preProcessor used to normalize the dataset based on Sergey
	 * Zagoruyko example 
	 * 
	 * <a href="https://github.com/szagoruyko/cifar.torch">https://github.com/szagoruyko/cifar.torch</a>
	 * 
	 * @see org.datavec.image.loader.CifarLoader
	 */
	CIFAR,
	/**
	 * Image loader for taking images and converting them to matrices
	 * 
	 * @see org.datavec.image.loader.ImageLoader
	 */
	DEFAULT,
	/**
	 * Loads LFW faces data transform. 
	 * 
	 * Customize the size of images by passing in preferred dimensions.
	 *
	 * DataSet 5749 different individuals 1680 people have two or more images in the
	 * database 4069 people have just a single image in the database available as
	 * 250 by 250 pixel JPEG images most images are in color, although a few are
	 * grayscale
	 * 
	 * @see org.bytedeco.opencv.spring.boot.nd4j.LFWLoader
	 */
	LFW,
	/**
	 * Uses JavaCV to load images. Allowed formats: bmp, gif, jpg, jpeg, jp2, pbm, pgm, ppm, pnm, png, tif, tiff, exr, webp
	 * 
	 * @see org.datavec.image.loader.NativeImageLoader
	 */
	NATIVE;

}
