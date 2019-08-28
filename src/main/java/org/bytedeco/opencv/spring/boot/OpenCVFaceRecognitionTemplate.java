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


import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.UUID;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.helper.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.CvHistogram;
import org.bytedeco.opencv.opencv_core.IplImage;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alibaba.fastjson.JSONObject;

/**
 * TODO
 * @author ： <a href="https://github.com/vindell">wandl</a>
 */
public class OpenCVFaceRecognitionTemplate {

	private static final Logger logger = LoggerFactory.getLogger(OpenCVFaceRecognitionTemplate.class);
	private CascadeClassifier faceDetector;
	private OpenCVFaceRecognitionProperties properties;
	
	public OpenCVFaceRecognitionTemplate(CascadeClassifier faceDetector,
			OpenCVFaceRecognitionProperties properties) {
		this.faceDetector = faceDetector;
	}

	public void smooth(String path) {
        IplImage image = opencv_imgcodecs.cvLoadImage(path);
        if (image != null) {
        	opencv_imgproc.cvSmooth(image, image);
        	opencv_imgcodecs.cvSaveImage(path, image);
        	opencv_core.cvReleaseImage(image);
        }
    }
	
	public JSONObject detect(String imagePath) {
		return detect(new File(imagePath));
	}
	
	public JSONObject detect(byte[] imageBytes, String filename) throws Exception {
		// 创建临时文件，因为boot打包后无法读取文件内的内容
    	File tempDir = new File(getProperties().getTemp());
    	if(!tempDir.exists()) {
    		tempDir.setReadable(true);
    		tempDir.setWritable(true);
    		tempDir.mkdir();
    	}
    	
    	File imageFile = new File(tempDir, UUID.randomUUID().toString() + "." + FilenameUtils.getExtension(filename));
    	try (InputStream source = new ByteArrayInputStream(imageBytes);){
    		FileUtils.copyInputStreamToFile(source, imageFile);
    		return detect(imageFile);
		}
    	
	}
	
	public JSONObject detect(File imageFile) {
		
		JSONObject result = new JSONObject();
		
		try {
			
			logger.info("人脸检测开始……");
		    
			if (imageFile == null || !imageFile.exists()) {
				result.put("error_code", 500);
				result.put("error_msg", "");
				return result;
	        }
			
			// 读取创建的图片tempFile
	        Mat image = Imgcodecs.imread(imageFile.getPath());
			// 进行人脸检测
	        MatOfRect faceDetections = new MatOfRect();
	        faceDetector.detectMultiScale(image, faceDetections);
	        
	        Rect[] rects = faceDetections.toArray();
	        if (rects == null || rects.length == 0 || rects.length > 1) {
	            return null;
	        }
	        
	        logger.info(String.format("检测到人脸： %s", rects.length));
	        
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
	public JSONObject match(String imagePath1, String imagePath2) {
		return match(new File(imagePath1), new File(imagePath2));
	}
	
	public JSONObject match(byte[] imageBytes1, byte[] imageBytes2, String filename) throws Exception {
		// 创建临时文件，因为boot打包后无法读取文件内的内容
    	File tempDir = new File(getProperties().getTemp());
    	if(!tempDir.exists()) {
    		tempDir.setReadable(true);
    		tempDir.setWritable(true);
    		tempDir.mkdir();
    	}
    	
    	File imageFile1 = new File(tempDir, UUID.randomUUID().toString() + "." + FilenameUtils.getExtension(filename));
    	File imageFile2 = new File(tempDir, UUID.randomUUID().toString() + "." + FilenameUtils.getExtension(filename));
    	try (InputStream source1 = new ByteArrayInputStream(imageBytes1);
    		 InputStream source2 = new ByteArrayInputStream(imageBytes2);){
    		FileUtils.copyInputStreamToFile(source1, imageFile1);
    		FileUtils.copyInputStreamToFile(source2, imageFile2);
    		return match(imageFile1, imageFile2);
		}
    	
	}
	
	public JSONObject match(File imageFile1, File imageFile2) {
		
		JSONObject result = new JSONObject();
		
		try {
			

			if (imageFile1 == null || !imageFile1.exists()) {
				result.put("error_code", 500);
				result.put("error_msg", "");
				return result;
	        }
			
			if (imageFile2 == null || !imageFile2.exists()) {
				result.put("error_code", 500);
				result.put("error_msg", "");
				return result;
	        }
		    
		    int lBins = 20;
	        int histSize[] = {lBins};

	        float vRanges[] = {0, 100};
	        float ranges[][] = {vRanges};

	        IplImage image1 = opencv_imgcodecs.cvLoadImage(imageFile1.getPath(), Imgcodecs.IMREAD_GRAYSCALE);
	        IplImage image2 = opencv_imgcodecs.cvLoadImage(imageFile2.getPath(), Imgcodecs.IMREAD_GRAYSCALE);

	        IplImage imageArr1[] = {image1};
	        IplImage imageArr2[] = {image2};

	        CvHistogram histogram1 = CvHistogram.create(1, histSize, Imgproc.HISTCMP_CORREL, ranges, 1);
	        CvHistogram histogram2 = CvHistogram.create(1, histSize, Imgproc.HISTCMP_CORREL, ranges, 1);

	        opencv_imgproc.cvCalcHist(imageArr1, histogram1, 0, null);
	        opencv_imgproc.cvCalcHist(imageArr2, histogram2, 0, null);

	        opencv_imgproc.cvNormalizeHist(histogram1, 100.0);
	        opencv_imgproc.cvNormalizeHist(histogram2, 100.0);

	        double score = opencv_imgproc.cvCompareHist(histogram1, histogram2, Imgproc.CV_COMP_CORREL);
	        
	        // 特征相似值
 			result.put("score", score);
		    
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
	public CascadeClassifier getFaceDetector() {
		return faceDetector;
	}
	
	public OpenCVFaceRecognitionProperties getProperties() {
		return properties;
	}

}
