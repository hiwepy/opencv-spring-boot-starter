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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

import org.bytedeco.opencv.spring.boot.dl4j.FaceNetSmallV2Model;
import org.bytedeco.opencv.spring.boot.nd4j.store.INDArrayStoreProvider;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.google.common.base.Optional;

/**
 * TODO
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 * @since 1.0.0
 */

public class Nd4jTemplate {

    private static final double THRESHOLD = 0.57;
	private FaceNetSmallV2Model faceNetSmallV2Model;
	
    private ComputationGraph computationGraph;
	private BaseImageLoader imageLoader;
	private INDArrayStoreProvider storeProvider;
	private int height;
	private int width;
	
	public Nd4jTemplate(ComputationGraph computationGraph, BaseImageLoader imageLoader, int height, int width) {
		this.computationGraph = computationGraph;
		this.imageLoader = imageLoader;
		this.height = height;
		this.width = width;
	}
    /**
     * <p>Face new.</p>
     * @param group the group
     * @param memberId the member id
     * @param imageBytes the image bytes
     * @throws IOException if an error occurs
     */
	
    public void faceNew(String group, String memberId, byte[] imageBytes) throws IOException {
        INDArray read = asMatrix(imageBytes);
        storeProvider.store(group, memberId, forwardPass(normalize(read)));
    }
    /**
     * <p>Face new.</p>
     * @param group the group
     * @param memberId the member id
     * @param imagePath the image path
     * @throws IOException if an error occurs
     */
    
    public void faceNew(String group, String memberId, File imagePath) throws IOException {
        INDArray read = asMatrix(imagePath);
        storeProvider.store(group, memberId, forwardPass(normalize(read)));
    }
    /**
     * <p>Face new.</p>
     * @param group the group
     * @param memberId the member id
     * @param imagePath the image path
     * @throws IOException if an error occurs
     */
    
    public void faceNew(String group, String memberId, String imagePath) throws IOException {
    	 INDArray read = asMatrix(imagePath);
    	 storeProvider.store(group, memberId, forwardPass(normalize(read)));
    }

    /**
     * 标准化
     * @author <a href="https://github.com/loong10k">Loong Wan</a>
     * @param read
     * @return
     */
    private static INDArray normalize(INDArray read) {
        return read.div(255.0);
    }
    /**
     * <p>Match.</p>
     * @param imageBytes1 the image bytes1
     * @param imageBytes2 the image bytes2
     * @throws IOException if an error occurs
     */

    public void match(byte[] imageBytes1, byte[] imageBytes2) throws IOException {
    	
        INDArray r1 = asMatrix(imageBytes1);
        INDArray r2 = asMatrix(imageBytes2);

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
    /**
     * <p>Search.</p>
     * @param group the group
     * @param memberId the member id
     * @return the string
     * @throws IOException if an error occurs
     */

    public String search(String group, String memberId) throws IOException {
        /*INDArray read = asMatrix(imagePath);
        INDArray encodings = forwardPass(normalize(read));
        double minDistance = Double.MAX_VALUE;
        String foundUser = "";
        
        INDArray value = storeProvider.get(group, memberId);
        double distance = Nd4jUtils.distance(value, encodings);
        System.out.println("distance of " + entry.getKey() + " with " + new File(imagePath).getName() + " is " + distance);
        if (distance < minDistance) {
            minDistance = distance;
            foundUser = entry.getKey();
        }
        if (minDistance > THRESHOLD) {
            foundUser = "Unknown user";
        }
        System.out.println(foundUser + " with distance " + minDistance);
        return foundUser;*/
    	return "";
    }
	/**
	 * <p>As matrix.</p>
	 * @param imageBytes the image bytes
	 * @return the i n d array
	 * @throws IOException if an error occurs
	 */
	
	public INDArray asMatrix(byte[] imageBytes) throws IOException {
		return asMatrix(new ByteArrayInputStream(imageBytes));
	}
	/**
	 * <p>As matrix.</p>
	 * @param imageStream the image stream
	 * @return the i n d array
	 * @throws IOException if an error occurs
	 */
	
	public INDArray asMatrix(InputStream imageStream) throws IOException {
		INDArray indArray = imageLoader.asMatrix(imageStream);
		return Nd4jUtils.transpose(indArray, height, width);
	}
	/**
	 * <p>As matrix.</p>
	 * @param imagePath the image path
	 * @return the i n d array
	 * @throws IOException if an error occurs
	 */

	public INDArray asMatrix(File imagePath) throws IOException {
        INDArray indArray = imageLoader.asMatrix(imagePath);
        return Nd4jUtils.transpose(indArray, height, width);
    }
	/**
	 * <p>As matrix.</p>
	 * @param imagePath the image path
	 * @return the i n d array
	 * @throws IOException if an error occurs
	 */
	
	public INDArray asMatrix(String imagePath) throws IOException {
        INDArray indArray = imageLoader.asMatrix(new File(imagePath));
        return Nd4jUtils.transpose(indArray, height, width);
    }
    /**
     * <p>Forward pass.</p>
     * @param indArray the ind array
     * @return the i n d array
     */
	
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
