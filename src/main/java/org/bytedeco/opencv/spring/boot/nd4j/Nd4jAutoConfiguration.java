package org.bytedeco.opencv.spring.boot.nd4j;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.bytedeco.opencv.spring.boot.OpenCVFaceRecognitionProperties;
import org.bytedeco.opencv.spring.boot.OpenCVFaceRecognitionTemplate;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.LFWLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

@Configuration
@ConditionalOnProperty(prefix = Nd4jProperties.PREFIX, value = "enabled", havingValue = "true")
@EnableConfigurationProperties({ Nd4jProperties.class, Nd4jImageLoaderProperties.class, Nd4jCifarLoaderProperties.class,
		Nd4jLFWLoaderProperties.class, Nd4NativeLoaderProperties.class })
public class Nd4jAutoConfiguration {

	@Bean
	public BaseImageLoader baseImageLoader(Nd4jProperties properties, 
			Nd4jCifarLoaderProperties cifarLoaderProperties, Nd4jImageLoaderProperties loaderProperties,
			Nd4jLFWLoaderProperties lfwLoaderProperties, Nd4NativeLoaderProperties nativeLoaderProperties) {
		
		BaseImageLoader imageLoader = null;
		switch (properties.getType()) {
			case CIFAR: {
				File fullPath = StringUtils.hasText(cifarLoaderProperties.getFullPath())
						? new File(cifarLoaderProperties.getFullPath()) : null;
				if (fullPath != null && fullPath.exists()) {
					imageLoader = new CifarLoader(true, fullPath);
				} else {
					imageLoader = new CifarLoader();
				}
			};break;
			case DEFAULT: {
				imageLoader = new ImageLoader(loaderProperties.getHeight(), loaderProperties.getWidth(),
						loaderProperties.getChannels(), loaderProperties.isCenterCropIfNeeded());
			};break;
			case LFW: {
				imageLoader = new LFWLoader(new long[] { lfwLoaderProperties.getHeight(), lfwLoaderProperties.getWidth(),
						lfwLoaderProperties.getChannels(), }, null, lfwLoaderProperties.isUseSubset());
			};break;
			default: {
				imageLoader = new NativeImageLoader(nativeLoaderProperties.getHeight(), nativeLoaderProperties.getWidth(),
						nativeLoaderProperties.getChannels(), nativeLoaderProperties.isCenterCropIfNeeded());
			};break;
		}
		return imageLoader;
	}
	
    @Bean
    public CascadeClassifier faceDetector(OpenCVFaceRecognitionProperties properties) throws IOException {
    	// 创建临时文件，因为boot打包后无法读取文件内的内容
    	File tempDir = new File(properties.getTemp());
    	if(!tempDir.exists()) {
    		tempDir.setReadable(true);
    		tempDir.setWritable(true);
    		tempDir.mkdir();
    	}
		File targetXmlFile = new File(tempDir, classifier.getFilename());
		FileUtils.copyInputStreamToFile(classifier.getInputStream(), targetXmlFile);
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		return new CascadeClassifier(targetXmlFile.getPath());
	}
    
	@Bean
	public OpenCVFaceRecognitionTemplate openCVFaceRecognitionTemplate(CascadeClassifier faceDetector,
			OpenCVFaceRecognitionProperties properties) {
		return new OpenCVFaceRecognitionTemplate(faceDetector, properties);
	}
	
}
