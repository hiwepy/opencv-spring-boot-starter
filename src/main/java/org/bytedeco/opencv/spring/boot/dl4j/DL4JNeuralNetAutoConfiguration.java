package org.bytedeco.opencv.spring.boot.dl4j;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.bytedeco.opencv.spring.boot.OpenCVFaceRecognitionProperties;
import org.bytedeco.opencv.spring.boot.OpenCVFaceRecognitionTemplate;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
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
 * <p>Auto-configuration for D L4 J Neural Net integration.</p>
 *
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 * @since 1.0.0
 */
public class DL4JNeuralNetAutoConfiguration {
	
	@Value("classpath:haarcascades/haarcascade_frontalface_alt.xml")
	private Resource classifier;
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
     * <p>Multi layer configuration.</p>
     * @param properties the properties
     * @return the multi layer configuration
     * @throws IOException if an error occurs
     */
	
    @Bean
    public MultiLayerConfiguration multiLayerConfiguration(OpenCVFaceRecognitionProperties properties) throws IOException {
		 MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		        .weightInit(WeightInit.XAVIER)
		        .activation(Activation.RELU)
		        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		        .updater(new Sgd(0.05))
		        // ... other hyperparameters
		        .list()
		        .build();
		return conf;
	}
    
	@Bean
	public OpenCVFaceRecognitionTemplate openCVFaceRecognitionTemplate(CascadeClassifier faceDetector,
			OpenCVFaceRecognitionProperties properties) {
		return new OpenCVFaceRecognitionTemplate(faceDetector, properties);
	}
	
}
