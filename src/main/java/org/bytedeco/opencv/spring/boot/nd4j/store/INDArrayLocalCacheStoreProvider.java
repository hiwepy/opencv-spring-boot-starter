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

import java.util.concurrent.TimeUnit;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.google.common.base.Optional;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.cache.RemovalListener;
import com.google.common.cache.RemovalNotification;

/**
 * TODO
 * @author 		： <a href="https://github.com/hiwepy">wandl</a>
 */

public class INDArrayLocalCacheStoreProvider implements INDArrayStoreProvider {

	/**
	 * INDArray 对象比较消耗内存，为了提高效率和防止内存溢出，这里只允许存储1000个对象，且10分钟过期；
	 */
	private final LoadingCache<String, Optional<INDArray>> INDARRAY_CACHES = CacheBuilder.newBuilder()
			// 设置并发级别为8，并发级别是指可以同时写缓存的线程数
			.concurrencyLevel(8)
			// 设置写缓存后600秒钟过期
			.expireAfterWrite(29, TimeUnit.DAYS)
			// 设置缓存容器的初始容量为10
			.initialCapacity(2)
			// 设置缓存最大容量为100，超过100之后就会按照LRU最近虽少使用算法来移除缓存项
			.maximumSize(10)
			// 设置要统计缓存的命中率
			.recordStats()
			// 设置缓存的移除通知
			.removalListener(new RemovalListener<String, Optional<INDArray>>() {
				@Override
				public void onRemoval(RemovalNotification<String, Optional<INDArray>> notification) {
					System.out.println(notification.getKey() + " was removed, cause is " + notification.getCause());
				}
			})
			// build方法中可以指定CacheLoader，在缓存不存在时通过CacheLoader的实现自动加载缓存
			.build(new CacheLoader<String, Optional<INDArray>>() {

				@Override
				public Optional<INDArray> load(String keySecret) throws Exception {
					return Optional.fromNullable(null);
				}
			});

	@Override
	public void store(String group, String memberId, INDArray ndarray) {
		 INDARRAY_CACHES.put(String.join("-", group, memberId), Optional.of(ndarray));
	}

	@Override
	public INDArray get(String group, String memberId) {
		Optional<INDArray> ndarray = INDARRAY_CACHES.getIfPresent(String.join("-", group, memberId));
		return ndarray.isPresent() ? ndarray.get() : null;
	}
	
}
