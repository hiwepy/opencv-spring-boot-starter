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
package org.bytedeco.opencv.spring.boot.dl4j;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.springframework.boot.context.properties.ConfigurationProperties;

import lombok.Getter;
import lombok.Setter;

@ConfigurationProperties(DL4JNeuralNetProperties.PREFIX)
@Getter
@Setter
public class DL4JNeuralNetProperties {

	public static final String PREFIX = "opencv.dl4j";

	// batch size: primarily used for conv nets. Will be reinforced if set.
	private boolean miniBatch = true;
	// number of line search iterations
	private int maxNumLineSearchIterations;
	
	private long seed;
	
	private OptimizationAlgorithm optimizationAlgo;
	
	// gradient keys used for ensuring order when getting and setting the gradient
	private List<String> variables = new ArrayList<>();
	
	// minimize or maximize objective
	private boolean minimize = true;

	// this field defines preOutput cache
	private CacheMode cacheMode = CacheMode.NONE;

	private DataType dataType = DataType.FLOAT; // Default to float for deserialization of legacy format nets
	
	// Counter for the number of parameter updates so far for this layer.
	// Note that this is only used for pretrain layers (AE, VAE) -
	// MultiLayerConfiguration and ComputationGraphConfiguration
	// contain counters for standard backprop training.
	// This is important for learning rate schedules, for example, and is stored
	// here to ensure it is persisted
	// for Spark and model serialization
	private int iterationCount = 0;

	// Counter for the number of epochs completed so far. Used for per-epoch
	// schedules
	private int epochCount = 0;

}
