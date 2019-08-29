package org.bytedeco.opencv.spring.boot.nd4j;

import lombok.extern.slf4j.Slf4j;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.spring.boot.dl4j.FaceNetSmallV2Model;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class FaceRecognition {

    private static final NativeImageLoader LOADER = new NativeImageLoader(96, 96, 3);
    
    private final HashMap<String, INDArray> memberEncodingsMap = new HashMap<>();

    

}
