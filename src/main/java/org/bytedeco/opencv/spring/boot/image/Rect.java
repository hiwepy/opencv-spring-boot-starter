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
 * 人脸位置信息类
 */
public class Rect {
    /**
     * 人脸矩形的最左边
     */
    public int left;
    /**
     * 人脸矩形的最上边
     */
    public int top;
    /**
     * 人脸矩形的最右边
     */
    public int right;
    /**
     * 人脸矩形的最下边
     */
    public int bottom;

    public Rect() {
    }

    /**
     * 根据传入的上下左右四个值创建一个人脸位置信息
     * @param left  人脸矩形的最左边
     * @param top   人脸矩形的最上边
     * @param right 人脸矩形的最右边
     * @param bottom  人脸矩形的最下边
     */
    public Rect(int left, int top, int right, int bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    /**
     * 根据传入的人脸矩形创建一个新的人脸矩形对象，新创建的对象为传入对象的深拷贝
     * @param r 人脸矩形信息对象
     */
    public Rect(Rect r) {
        if (r == null) {
            this.left = this.top = this.right = this.bottom = 0;
        } else {
            this.left = r.left;
            this.top = r.top;
            this.right = r.right;
            this.bottom = r.bottom;
        }

    }

    /**
     *
     * @return  格式化的人脸位置信息
     */
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("com.arcsoft.face.Rect(");
        sb.append(this.left);
        sb.append(", ");
        sb.append(this.top);
        sb.append(" - ");
        sb.append(this.right);
        sb.append(", ");
        sb.append(this.bottom);
        sb.append(")");
        return sb.toString();
    }

    public int getLeft() {
        return left;
    }

    public void setLeft(int left) {
        this.left = left;
    }

    public int getTop() {
        return top;
    }

    public void setTop(int top) {
        this.top = top;
    }

    public int getRight() {
        return right;
    }

    public void setRight(int right) {
        this.right = right;
    }

    public int getBottom() {
        return bottom;
    }

    public void setBottom(int bottom) {
        this.bottom = bottom;
    }
}
