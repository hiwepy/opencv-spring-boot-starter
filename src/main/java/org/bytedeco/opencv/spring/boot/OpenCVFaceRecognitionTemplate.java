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
package org.bytedeco.opencv.spring.boot;

import static org.opencv.imgproc.Imgproc.CV_COMP_CORREL;

import java.io.File;

import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.helper.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.CvHistogram;
import org.bytedeco.opencv.opencv_core.IplImage;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.alibaba.fastjson.JSONObject;
import com.baidu.aip.util.Base64Util;

/**
 * TODO
 * 
 * @author ： <a href="https://github.com/vindell">wandl</a>
 */

public class OpenCVFaceRecognitionTemplate {

	private CascadeClassifier faceDetector;
	private OpenCVFaceRecognitionProperties properties;
	private static final Logger logger = LoggerFactory.getLogger(OpenCVFaceRecognitionProvider.class);
    public static final double MATCH_PERCENT = 0.5;
    public static final int LOST_PERSON = 1;
    public static final int FIND_PERSON = 2;
	
	public OpenCVFaceRecognitionTemplate(CascadeClassifier faceDetector, OpenCVFaceRecognitionProperties properties) {
		this.faceDetector = faceDetector;
		this.properties = properties;
		 System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public void smooth(String path) {
        IplImage image = opencv_imgcodecs.cvLoadImage(path);
        if (image != null) {
        	cvSmooth(image, image);
        	opencv_imgcodecs.cvSaveImage(path, image);
        	cvReleaseImage(image);
        }
    }
	

    public double comparePictures(String imagePath1, String imagePath2) {

        int lBins = 20;
        int histSize[] = {lBins};

        float vRanges[] = {0, 100};
        float ranges[][] = {vRanges};

        IplImage image1 = opencv_imgcodecs.cvLoadImage(imagePath1, Imgcodecs.IMREAD_GRAYSCALE);
        IplImage image2 = opencv_imgcodecs.cvLoadImage(imagePath2, Imgcodecs.IMREAD_GRAYSCALE);

        IplImage imageArr1[] = {image1};
        IplImage imageArr2[] = {image2};

        CvHistogram histogram1 = CvHistogram.create(1, histSize, Imgproc.HISTCMP_CORREL, ranges, 1);
        CvHistogram histogram2 = CvHistogram.create(1, histSize, Imgproc.HISTCMP_CORREL, ranges, 1);

        cvCalcHist(imageArr1, histogram1, 0, null);
        cvCalcHist(imageArr2, histogram2, 0, null);

        cvNormalizeHist(histogram1, 100.0);
        cvNormalizeHist(histogram2, 100.0);

        return cvCompareHist(histogram1, histogram2, CV_COMP_CORREL);
    }
    
	public JSONObject detect(byte[] imageBytes) {
		return detect(imageBytes, FaceType.LIVE);
	}
	
	public JSONObject detect(byte[] imageBytes, FaceType face_type) {
		return detect(imageBytes, face_type, FaceLiveness.NONE);
	}
	
	/**
	 * 人脸检测与属性分析： https://ai.baidu.com/docs#/Face-Detect-V3/top
	 * 
	 * @param imageBytes       图片字节码：现支持PNG、JPG、JPEG、BMP，不支持GIF图片
	 * @param face_type        人脸的类型：LIVE表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
	 *                         IDCARD表示身份证芯片照：二代身份证内置芯片中的人像照片
	 *                         WATERMARK表示带水印证件照：一般为带水印的小图，如公安网小图
	 *                         CERT表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片 默认LIVE
	 * @param liveness		        活体控制 检测结果中不符合要求的人脸会被过滤 NONE: 不进行控制 LOW:较低的活体要求(高通过率
	 *                         低攻击拒绝率) NORMAL: 一般的活体要求(平衡的攻击拒绝率, 通过率) HIGH:
	 *                         较高的活体要求(高攻击拒绝率 低通过率) 默认NONE
	 * @author ： <a href="https://github.com/vindell">wandl</a>
	 * @return
	 */
	public JSONObject detect(byte[] imageBytes, FaceType face_type, FaceLiveness liveness) {
		try {
			String imgStr = Base64Util.encode(imageBytes);
			return detect(imgStr, face_type, liveness);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public JSONObject detect(String imageBase64) {
		return detect(imageBase64, FaceType.LIVE);
	}
	
	public JSONObject detect(String imageBase64, FaceType face_type) {
		return detect(imageBase64, face_type, FaceLiveness.NONE);
	}
	
	/**
	 * 人脸检测与属性分析： https://ai.baidu.com/docs#/Face-Detect-V3/top
	 * 
	 * @param imageBase64      Base64编码：请求的图片需经过Base64编码，图片的base64编码指将图片数据编码成一串字符串，使用该字符串代替图像地址。
	 *                         您可以首先得到图片的二进制，然后用Base64格式编码即可。需要注意的是，图片的base64编码是不包含图片头的，如data:image/jpg;base64,
	 *                         图片格式：现支持PNG、JPG、JPEG、BMP，不支持GIF图片
	 * @param face_type        人脸的类型：LIVE表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
	 *                         IDCARD表示身份证芯片照：二代身份证内置芯片中的人像照片
	 *                         WATERMARK表示带水印证件照：一般为带水印的小图，如公安网小图
	 *                         CERT表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片 默认LIVE
	 * @param liveness 活体控制 检测结果中不符合要求的人脸会被过滤 NONE: 不进行控制 LOW:较低的活体要求(高通过率
	 *                         低攻击拒绝率) NORMAL: 一般的活体要求(平衡的攻击拒绝率, 通过率) HIGH:
	 *                         较高的活体要求(高攻击拒绝率 低通过率) 默认NONE
	 * @author ： <a href="https://github.com/vindell">wandl</a>
	 * @return
	 */
	public JSONObject detect(String imageBase64, FaceType face_type, FaceLiveness liveness) {
		
		JSONObject result = new JSONObject();
		
		try {
			
			logger.info("人脸检测开始……");
		    
			if (source == null) {
	           
	        }
			
			File tempDir = new File(faceRecognitionProperties.getTemp());
	    	if(!tempDir.exists()) {
	    		tempDir.setReadable(true);
	    		tempDir.setWritable(true);
	    		tempDir.mkdir();
	    	}
			 
	        // 创建图片tempFile
	        File tempFile = new File(tempDir,  source.getOriginalFilename());
	        source.transferTo(tempFile);

			// 读取创建的图片tempFile
	        Mat image = Imgcodecs.imread(tempFile.getPath());
			// 进行人脸检测
	        MatOfRect faceDetections = new MatOfRect();
	        faceDetector.detectMultiScale(image, faceDetections);
	        logger.info(String.format("检测到人脸： %s", faceDetections.toArray().length));
	        
	        Rect[] rects = faceDetections.toArray();
	        if (rects == null || rects.length == 0 || rects.length > 1) {
	            return null;
	        }
	        
	        //	         在每一个识别出来的人脸周围画出一个方框
	        Rect rect = rects[0];
	        Imgproc.rectangle(image, new Point(rect.x - 2, rect.y - 2),
	                new Point(rect.x + rect.width, rect.y + rect.height),
	                new Scalar(0, 255, 0));
	        String outFile =
	                new ClassPathResource(1 == LOST_PERSON ? "static/upload/lostfaceimages" : "static/upload/findfaceimages").getFile().getPath() + "/" + saveImageName;
	        Imgcodecs.imwrite(outFile, image);
	        logger.info(String.format("人脸识别成功，人脸图片文件为： %s", outFile));
		        
			
			return JSONObject.parseObject(result);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
	public JSONObject match(byte[] imageBytes_1, byte[] imageBytes_2) {
		return match(imageBytes_1, imageBytes_2, FaceType.LIVE);
	}
	
	public JSONObject match(byte[] imageBytes_1, byte[] imageBytes_2, FaceType face_type) {
		return match(imageBytes_1, imageBytes_2, face_type, FaceQuality.LOW);
	}
	
	public JSONObject match(byte[] imageBytes_1, byte[] imageBytes_2, FaceType face_type, FaceQuality quality) {
		return match(imageBytes_1, imageBytes_2, face_type, quality, FaceLiveness.NORMAL);
	}
	
	/**
	 * 人脸对比: https://ai.baidu.com/docs#/Face-Match-V3/top
	 * 
	 * @author ： <a href="https://github.com/vindell">wandl</a>
	 * @param imageBytes_1     图片字节码：现支持PNG、JPG、JPEG、BMP，不支持GIF图片
	 * @param imageBytes_2     图片字节码：现支持PNG、JPG、JPEG、BMP，不支持GIF图片
	 * @param face_type        人脸的类型：LIVE表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
	 *                         IDCARD表示身份证芯片照：二代身份证内置芯片中的人像照片
	 *                         WATERMARK表示带水印证件照：一般为带水印的小图，如公安网小图
	 *                         CERT表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片 默认LIVE
	 * @param quality  图片质量控制 NONE: 不进行控制 LOW:较低的质量要求 NORMAL: 一般的质量要求 HIGH:
	 *                         较高的质量要求 默认 NONE 若图片质量不满足要求，则返回结果中会提示质量检测失败
	 * @param liveness 活体控制 检测结果中不符合要求的人脸会被过滤 NONE: 不进行控制 LOW:较低的活体要求(高通过率
	 *                         低攻击拒绝率) NORMAL: 一般的活体要求(平衡的攻击拒绝率, 通过率) HIGH:
	 *                         较高的活体要求(高攻击拒绝率 低通过率) 默认NONE
	 * @return
	 */
	public JSONObject match(byte[] imageBytes_1, byte[] imageBytes_2, FaceType face_type, FaceQuality quality, FaceLiveness liveness) {
		String imageBase64_1 = Base64Util.encode(imageBytes_1);
		String imageBase64_2 = Base64Util.encode(imageBytes_2);
		return match(imageBase64_1, imageBase64_2, face_type, quality, liveness);
	}

	public JSONObject match(String imageBase64_1, String imageBase64_2) {
		return match(imageBase64_1, imageBase64_2, FaceType.LIVE);
	}
	
	public JSONObject match(String imageBase64_1, String imageBase64_2, FaceType face_type) {
		return match(imageBase64_1, imageBase64_2, face_type, FaceQuality.LOW);
	}
	
	public JSONObject match(String imageBase64_1, String imageBase64_2, FaceType face_type, FaceQuality quality) {
		return match(imageBase64_1, imageBase64_2, face_type, quality,  FaceLiveness.NORMAL);
	}
	
	/**
	 * 人脸对比: https://ai.baidu.com/docs#/Face-Match-V3/top
	 * 
	 * @param imageBase64_1    Base64编码：请求的图片需经过Base64编码，图片的base64编码指将图片数据编码成一串字符串，使用该字符串代替图像地址。
	 *                         您可以首先得到图片的二进制，然后用Base64格式编码即可。需要注意的是，图片的base64编码是不包含图片头的，如data:image/jpg;base64,
	 *                         图片格式：现支持PNG、JPG、JPEG、BMP，不支持GIF图片
	 * @param imageBase64_2    Base64编码：请求的图片需经过Base64编码，图片的base64编码指将图片数据编码成一串字符串，使用该字符串代替图像地址。
	 *                         您可以首先得到图片的二进制，然后用Base64格式编码即可。需要注意的是，图片的base64编码是不包含图片头的，如data:image/jpg;base64,
	 *                         图片格式：现支持PNG、JPG、JPEG、BMP，不支持GIF图片
	 * @param face_type        人脸的类型：LIVE表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
	 *                         IDCARD表示身份证芯片照：二代身份证内置芯片中的人像照片
	 *                         WATERMARK表示带水印证件照：一般为带水印的小图，如公安网小图
	 *                         CERT表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片 默认LIVE
	 * @param quality  图片质量控制； NONE: 不进行控制； LOW:较低的质量要求； NORMAL: 一般的质量要求；
	 *                         HIGH: 较高的质量要求； 默认 NONE； 若图片质量不满足要求，则返回结果中会提示质量检测失败
	 * @param liveness 活体控制 检测结果中不符合要求的人脸会被过滤； NONE: 不进行控制 ；LOW:较低的活体要求(高通过率
	 *                         低攻击拒绝率)； NORMAL: 一般的活体要求(平衡的攻击拒绝率, 通过率)； HIGH:
	 *                         较高的活体要求(高攻击拒绝率 低通过率)； 默认NONE
	 * @author ： <a href="https://github.com/vindell">wandl</a>
	 * @return
	 */
	public JSONObject match(String imageBase64_1, String imageBase64_2, FaceType face_type, FaceQuality quality, FaceLiveness liveness) {
		
		JSONObject result = new JSONObject();
		
		try {

		    int l_bins = 20;
		    int hist_size[] = {l_bins};

		    float v_ranges[] = {0, 100};
		    float ranges[][] = {v_ranges};

		    IplImage Image1 = opencv_imgcodecs.cvLoadImage("/FaceDetect/" + file1 + ".jpg", Imgcodecs.IMREAD_GRAYSCALE);
		    IplImage Image2 = opencv_imgcodecs.cvLoadImage("/FaceDetect/" + file2 + ".jpg", Imgcodecs.IMREAD_GRAYSCALE);

		    IplImage imageArr1[] = {Image1};
		    IplImage imageArr2[] = {Image2};

		    CvHistogram Histogram1 = CvHistogram.create(1, hist_size, Imgproc.HISTCMP_CORREL, ranges, 1);
		    CvHistogram Histogram2 = CvHistogram.create(1, hist_size, Imgproc.HISTCMP_CORREL, ranges, 1);

		    opencv_imgproc.cvCalcHist(imageArr1, Histogram1, 0, null);
		    opencv_imgproc.cvCalcHist(imageArr2, Histogram2, 0, null);

		    opencv_imgproc.cvNormalizeHist(Histogram1, 100.0);
		    opencv_imgproc.cvNormalizeHist(Histogram2, 100.0);

		    opencv_imgproc.cvCompareHist(Histogram1, Histogram2, Imgproc.HISTCMP_CORREL);
			
		    return null;
		     
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

}
