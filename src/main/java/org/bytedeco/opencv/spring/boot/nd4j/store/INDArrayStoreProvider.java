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
package org.bytedeco.opencv.spring.boot.nd4j.store;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * TODO
 * 
 * @author ： <a href="https://github.com/hiwepy">wandl</a>
 */
public interface INDArrayStoreProvider {

	void store(String group, String memberId, INDArray ndarray);
	
	INDArray get(String group, String memberId);
	
}
