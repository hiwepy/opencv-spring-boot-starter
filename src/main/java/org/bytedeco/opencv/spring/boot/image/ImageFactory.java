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
package org.bytedeco.opencv.spring.boot.image;

import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;


public class ImageFactory {


    public static ImageInfo getRGBData(File file) {
        if (file == null)
            return null;
        ImageInfo imageInfo;
        try {
            //将图片文件加载到内存缓冲区
            BufferedImage image = ImageIO.read(file);
            imageInfo = bufferedImage2ImageInfo(image);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        return imageInfo;
    }

    public static ImageInfo getGrayData(File file) {
        if (file == null)
            return null;
        ImageInfo imageInfo;
        try {
            //将图片文件加载到内存缓冲区
            BufferedImage image = ImageIO.read(file);
            imageInfo = bufferedImage2GrayImageInfo(image);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        return imageInfo;
    }

    public static ImageInfo getRGBData(byte[] bytes) {
        if (bytes == null)
            return null;
        return getRGBData(new ByteArrayInputStream(bytes));
    }

    public static ImageInfo getGrayData(byte[] bytes) {
        if (bytes == null)
            return null;
        return getGrayData(new ByteArrayInputStream(bytes));
    }


    public static ImageInfo getRGBData(InputStream input) {
        if (input == null)
            return null;
        ImageInfo imageInfo;
        try {
            BufferedImage image = ImageIO.read(input);
            if (image == null) {
                return null;
            }
            imageInfo = bufferedImage2ImageInfo(image);

        } catch (IOException e) {
            return null;
        } finally {
            try {
                input.close();
            } catch (IOException e) {
            }

        }
        return imageInfo;
    }

    public static ImageInfo getGrayData(InputStream input) {
        if (input == null)
            return null;
        ImageInfo imageInfo;
        try {
            BufferedImage image = ImageIO.read(input);
            if (image == null) {
                return null;
            }
            imageInfo = bufferedImage2GrayImageInfo(image);

        } catch (IOException e) {
            return null;
        } finally {
            try {
                input.close();
            } catch (IOException e) {
            }

        }
        return imageInfo;
    }


    public static ImageInfo bufferedImage2ImageInfo(BufferedImage image) {
        ImageInfo imageInfo = new ImageInfo();
        int width = image.getWidth();
        int height = image.getHeight();
        // 使图片居中
        width = width & (~3);
        height = height & (~3);
        imageInfo.setWidth(width);
        imageInfo.setHeight(height);
        //根据原图片信息新建一个图片缓冲区
        BufferedImage resultImage = new BufferedImage(width, height, image.getType());
        //得到原图的rgb像素矩阵
        int[] rgb = image.getRGB(0, 0, width, height, null, 0, width);
        //将像素矩阵 绘制到新的图片缓冲区中
        resultImage.setRGB(0, 0, width, height, rgb, 0, width);
        //进行数据格式化为可用数据
        BufferedImage dstImage = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        if (resultImage.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_LINEAR_RGB);
            ColorConvertOp colorConvertOp = new ColorConvertOp(cs, dstImage.createGraphics().getRenderingHints());
            colorConvertOp.filter(resultImage, dstImage);
        } else {
            dstImage = resultImage;
        }
        //获取rgb数据
        imageInfo.setImageFormat(ImageFormat.CP_PAF_BGR24);
        imageInfo.setImageData(((DataBufferByte) (dstImage.getRaster().getDataBuffer())).getData());
        return imageInfo;
    }

    public static ImageInfo bufferedImage2GrayImageInfo(BufferedImage image) {
        ImageInfo imageInfo = new ImageInfo();
        int width = image.getWidth();
        int height = image.getHeight();
        // 使图片居中
        width = width & (~3);
        height = height & (~3);
        imageInfo.setWidth(width);
        imageInfo.setHeight(height);

        int[] rgb = image.getRGB(0, 0, width, height, null, 0, width);
        byte[] bytes = rgbToGray(rgb, width, height);
        imageInfo.setImageFormat(ImageFormat.CP_PAF_GRAY);
        imageInfo.setImageData(bytes);
        return imageInfo;
    }


    private static byte[] rgbToGray(int[] argb, int width, int height) {

        int yIndex = 0;
        int index = 0;
        byte[] gray = new byte[width * height];
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                int R = (argb[index] & 0xFF0000) >> 16;
                int G = (argb[index] & 0x00FF00) >> 8;
                int B = argb[index] & 0x0000FF;
                int Y = (66 * R + 129 * G + 25 * B + 128 >> 8) + 16;
//                int U = (-38 * R - 74 * G + 112 * B + 128 >> 8) + 128;
//                int V = (112 * R - 94 * G - 18 * B + 128 >> 8) + 128;
                gray[yIndex++] = (byte) (Y < 0 ? 0 : (Y > 255 ? 255 : Y));
                ++index;
            }
        }
        return gray;
    }


    /**
     * 将图像中需要截取的Rect向外扩张一倍，若扩张一倍会溢出，则扩张到边界，若Rect已溢出，则收缩到边界
     *
     * @param width   图像宽度
     * @param height  图像高度
     * @param srcRect 原Rect
     * @return 调整后的Rect
     */
    public static Rect getBestRect(int width, int height, Rect srcRect) {
        if (srcRect == null) {
            return null;
        }
        Rect rect = new Rect(srcRect);
        int maxOverFlow = Math.max(-rect.left, Math.min(-rect.top, Math.min(width - rect.right, height - rect.bottom)));
        // 原rect边界已溢出宽高的情况
        if (maxOverFlow > 0) {
            rect.left += maxOverFlow;
            rect.top += maxOverFlow;
            rect.right -= maxOverFlow;
            rect.bottom -= maxOverFlow;
            return rect;
        }
        // 原rect边界未溢出宽高的情况
        int padding = (rect.bottom - rect.top) / 2;
        // 若以此padding扩张rect会溢出，取最大padding为四个边距的最小值
        if (!(rect.left - padding > 0
                && rect.right + padding < width
                && rect.top - padding > 0
                && rect.bottom + padding < height)) {
            padding = Math.min(Math.min(Math.min(rect.left, width - rect.right), height - rect.bottom), rect.top);
        }
        rect.left -= padding;
        rect.top -= padding;
        rect.right += padding;
        rect.bottom += padding;
        return rect;
    }

}
