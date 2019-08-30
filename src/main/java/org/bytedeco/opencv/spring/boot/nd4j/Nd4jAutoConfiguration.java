package org.bytedeco.opencv.spring.boot.nd4j;

import java.io.File;

import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.LFWLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
	public BaseImageLoader imageLoader(Nd4jProperties properties, 
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
	public Nd4jTemplate Nd4jTemplate(ComputationGraph computationGraph, BaseImageLoader imageLoader) {
		return new Nd4jTemplate(computationGraph, imageLoader,1,1);
	}
	
}
