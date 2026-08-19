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
package org.bytedeco.opencv.spring.boot;

import org.apache.commons.lang3.SystemUtils;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(OpenCVFaceRecognitionProperties.PREFIX)
/**
 * <p>Configuration properties for Open C V Face Recognition.</p>
 *
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 * @since 1.0.0
 */
public class OpenCVFaceRecognitionProperties {

	public static final String PREFIX = "opencv.face";

	/**
	 * 	Enable OpenCV Face Recognition.
	 */
	private boolean enabled = false;
	/**
	 * 人脸识别图片临时目录
	 */
	private String temp = SystemUtils.getUserDir().getAbsolutePath();
	/**
	 * <p>Is enabled.</p>
	 * @return the boolean
	 */
	
	public boolean isEnabled() {
		return enabled;
	}
	/** Sets the enabled. */

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}
	/** Gets the temp. */

	public String getTemp() {
		return temp;
	}
	/** Sets the temp. */

	public void setTemp(String temp) {
		this.temp = temp;
	}

}
