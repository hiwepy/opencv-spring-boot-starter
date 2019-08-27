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

import org.apache.commons.lang3.SystemUtils;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(OpenCVFaceRecognitionProperties.PREFIX)
public class OpenCVFaceRecognitionProperties {

	public static final String PREFIX = "opencv.face";

	/**
	 * 	Enable Baidu Face Recognition.
	 */
	private boolean enabled = false;
	/**
	 * 	官网获取的 API Key（百度云应用的AK）
	 */
	private String clientId;
	/**
	 * 	官网获取的 Secret Key（百度云应用的SK）
	 */
	private String clientSecret;
	/**
	 * 人脸识别图片临时目录
	 */
	private String temp = SystemUtils.getUserDir().getAbsolutePath();
	
	public boolean isEnabled() {
		return enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	public String getClientId() {
		return clientId;
	}

	public void setClientId(String clientId) {
		this.clientId = clientId;
	}

	public String getClientSecret() {
		return clientSecret;
	}

	public void setClientSecret(String clientSecret) {
		this.clientSecret = clientSecret;
	}

	public String getTemp() {
		return temp;
	}

	public void setTemp(String temp) {
		this.temp = temp;
	}

}
