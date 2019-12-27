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
package org.bytedeco.opencv.spring.boot.nd4j;

import org.datavec.image.loader.LFWLoader;
import org.springframework.boot.context.properties.ConfigurationProperties;

import lombok.Getter;
import lombok.Setter;

@ConfigurationProperties(Nd4jLFWLoaderProperties.PREFIX)
@Getter
@Setter
public class Nd4jLFWLoaderProperties {

	public static final String PREFIX = "opencv.nd4j.loader.lfw";
	/**
	 * the height to load
	 */
    private long height = LFWLoader.HEIGHT;
    /**
     * the width to load
     */
    private long width = LFWLoader.WIDTH;
    /**
     * the number of channels for the image*
     */
    private long channels = LFWLoader.CHANNELS;
    /**
     */
    private boolean useSubset = false;
    
}
