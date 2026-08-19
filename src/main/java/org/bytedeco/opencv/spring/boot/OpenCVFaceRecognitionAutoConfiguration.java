package org.bytedeco.opencv.spring.boot;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.bytedeco.opencv.spring.boot.dl4j.FaceNetSmallV2Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

@Configuration
@ConditionalOnProperty(prefix = OpenCVFaceRecognitionProperties.PREFIX, value = "enabled", havingValue = "true")
@EnableConfigurationProperties({ OpenCVFaceRecognitionProperties.class })
/**
 * <p>Auto-configuration for Open C V Face Recognition integration.</p>
 *
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 * @since 1.0.0
 */
public class OpenCVFaceRecognitionAutoConfiguration {
	
	@Value("classpath:haarcascades/haarcascade_frontalface_alt.xml")
	private Resource classifier;
	
	static {
		Loader.load(opencv_java.class);
		//new opencv_java();
		
		
	}
	/**
	 * <p>Face net small v2 model.</p>
	 * @return the face net small v2 model
	 */
	
	@Bean
	public FaceNetSmallV2Model faceNetSmallV2Model() {
		return new FaceNetSmallV2Model();
	}
	/**
	 * <p>Computation graph.</p>
	 * @param faceNetSmallV2Model the face net small v2 model
	 * @return the computation graph
	 * @throws Exception if an error occurs
	 */

	@Bean
	public ComputationGraph computationGraph(FaceNetSmallV2Model faceNetSmallV2Model) throws Exception {

		ComputationGraph computationGraph = faceNetSmallV2Model.init();
		System.out.println(computationGraph.summary());
		
		return computationGraph;
	}
    /**
     * <p>Face detector.</p>
     * @param properties the properties
     * @return the cascade classifier
     * @throws IOException if an error occurs
     */
	
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
