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
package org.bytedeco.opencv.spring.boot.nd4j;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.bytedeco.opencv.spring.boot.dl4j.FaceNetSmallV2Model;
import org.bytedeco.opencv.spring.boot.image.ImageFactory;
import org.bytedeco.opencv.spring.boot.image.ImageInfo;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.alibaba.fastjson.JSONObject;
import com.google.common.base.Optional;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.cache.RemovalListener;
import com.google.common.cache.RemovalNotification;

/**
 * TODO
 * @author 		： <a href="https://github.com/vindell">wandl</a>
 */

public class Nd4jTemplate {

    private static final double THRESHOLD = 0.57;
	private FaceNetSmallV2Model faceNetSmallV2Model;
    private ComputationGraph computationGraph;
	private BaseImageLoader imageLoader;
	private int height;
	private int width;
	
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
					
					
					
					JSONObject key = JSONObject.parseObject(keySecret);
					String token = AuthClient.getAuth(key.getString("clientId"), key.getString("clientSecret"));
					return Optional.fromNullable(token);
				}
			});
	
	
    public void faceNew(String memberId, String imagePath) throws IOException {
    	
    	
        INDArray read = read(imagePath);
        memberEncodingsMap.put(memberId, forwardPass(normalize(read)));
    }

    /**
     * 标准化
     * @author 		： <a href="https://github.com/vindell">wandl</a>
     * @param read
     * @return
     */
    private static INDArray normalize(INDArray read) {
        return read.div(255.0);
    }

    public void match(byte[] imageBytes1, byte[] imageBytes2) throws IOException {
    	
        INDArray r1 = read(imageBytes1);
        INDArray r2 = read(imageBytes2);

        INDArray e1 = forwardPass(normalize(r1));
        INDArray e2 = forwardPass(normalize(r2));

        double dis = Nd4jUtils.distance(e1, e2);
        System.out.println("distance is : " + dis);
        if (dis < 0.45) {
            System.out.println("match");
        } else {
            System.out.println("dismatch");
        }
    }

    public String search(String imagePath) throws IOException {
        INDArray read = read(imagePath);
        INDArray encodings = forwardPass(normalize(read));
        double minDistance = Double.MAX_VALUE;
        String foundUser = "";
        
        for (Map.Entry<String, INDArray> entry : memberEncodingsMap.entrySet()) {
            INDArray value = entry.getValue();
            double distance = Nd4jUtils.distance(value, encodings);
            System.out.println("distance of " + entry.getKey() + " with " + new File(imagePath).getName() + " is " + distance);
            if (distance < minDistance) {
                minDistance = distance;
                foundUser = entry.getKey();
            }
        }
        if (minDistance > THRESHOLD) {
            foundUser = "Unknown user";
        }
        System.out.println(foundUser + " with distance " + minDistance);
        return foundUser;
    }
	
	public INDArray read(byte[] imageBytes) throws IOException {
		ImageInfo imageInfo = ImageFactory.getRGBData(imageBytes);
		return read(new ByteArrayInputStream(imageInfo.getImageData()), imageInfo.getHeight(), imageInfo.getWidth());
	}
	
	public INDArray read(InputStream inputStream, int height, int width) throws IOException {
		INDArray indArray = imageLoader.asMatrix(inputStream);
		return Nd4jUtils.transpose(indArray, height, width);
	}
	
	public INDArray read(String pathname, int height, int width) throws IOException {
        INDArray indArray = imageLoader.asMatrix(new File(pathname));
        return Nd4jUtils.transpose(indArray, height, width);
    }
	
    private INDArray forwardPass(INDArray indArray) {
    	
        Map<String, INDArray> output = computationGraph.feedForward(indArray, false);
        GraphVertex embeddings = computationGraph.getVertex("encodings");
        INDArray dense = output.get("dense");
        embeddings.setInputs(dense);
        INDArray embeddingValues = embeddings.doForward(false, LayerWorkspaceMgr.builder().defaultNoWorkspace().build());
        System.out.println("dense =                 " + dense);
        System.out.println("encodingsValues =                 " + embeddingValues);
        return embeddingValues;
    }

}
