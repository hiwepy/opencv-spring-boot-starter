/*
 * Copyright (c) 2018, vindell (https://github.com/vindell).
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


/**
 * 图像格式
 */
public enum ImageFormat {
    /**
     * 图像格式，NV21格式，8-bit Y 通道，8-bit 2x2 采样 V 与 U 分量交织通道
     */
    CP_PAF_NV21(2050),
    /**
     * 图像格式，NV12格式，8-bit Y 通道，8-bit 2x2 采样 U 与 V 分量交织通道
     */
    CP_PAF_NV12(2049),
    /**
     * 图像格式，I420格式，8-bit Y 通道，8-bit 2x2 采样 U 通道，8-bit 2x2 采样 V 通道
     */
    CP_PAF_I420(1537),
    /**
     * 图像格式，YUYV格式，YUV 分量交织，V 与 U 分量 2x1 采样，按 Y0, U0, Y1, V0 字节序排布
     */
    CP_PAF_YUYV(1281),
    /**
     * 图像格式，BGR格式，RGB 分量交织，按 B, G, R, B 字节序排布
     */
    CP_PAF_BGR24(513),
    /**
     * 图像格式，BGR格式，RGB 分量交织，按 B, G, R, B 字节序排布
     */
    CP_PAF_GRAY(1793),

    /**
     * 图像格式，BGR格式，RGB 分量交织，按 B, G, R, B 字节序排布
     */
    CP_PAF_DEPTH_U16(3074);


    private int value;

    ImageFormat(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
