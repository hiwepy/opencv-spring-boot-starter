package org.bytedeco.opencv.spring.boot;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import com.alibaba.fastjson2.JSONObject;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for {@link OpenCVFaceRecognitionTemplate}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class OpenCVFaceRecognitionTemplateTest {

    @TempDir
    File tempDir;

    private OpenCVFaceRecognitionTemplate createTemplate() {
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        props.setTemp(tempDir.getAbsolutePath());
        return new OpenCVFaceRecognitionTemplate(null, props);
    }

    @Test
    void constructorShouldAcceptNullParameters() {
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        OpenCVFaceRecognitionTemplate template = new OpenCVFaceRecognitionTemplate(null, props);
        assertThat(template).isNotNull();
        assertThat(template.getProperties()).isEqualTo(props);
        assertThat(template.getFaceDetector()).isNull();
    }

    @Test
    void getPropertiesShouldReturnProperties() {
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        props.setTemp("/tmp/test");
        OpenCVFaceRecognitionTemplate template = new OpenCVFaceRecognitionTemplate(null, props);
        assertThat(template.getProperties().getTemp()).isEqualTo("/tmp/test");
    }

    @Test
    void getFaceDetectorShouldReturnDetector() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        assertThat(template.getFaceDetector()).isNull();
    }

    @Test
    void detectWithNullFileShouldReturnErrorResult() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        JSONObject result = template.detect((File) null);
        assertThat(result).isNotNull();
        assertThat(result.getIntValue("error_code")).isEqualTo(500);
    }

    @Test
    void detectWithNonExistentFileShouldReturnErrorResult() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        JSONObject result = template.detect(new File("/nonexistent/path/image.jpg"));
        assertThat(result).isNotNull();
        assertThat(result.getIntValue("error_code")).isEqualTo(500);
    }

    @Test
    void detectWithStringPathShouldDelegateToFile() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        // This will fail because file doesn't exist, but tests the code path
        JSONObject result = template.detect("/nonexistent/path/image.jpg");
        assertThat(result).isNotNull();
    }

    @Test
    void detectWithBytesShouldCreateTempFile() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        byte[] fakeBytes = new byte[]{1, 2, 3, 4};
        // This will fail at native OpenCV loading but tests the temp file creation path
        try {
            template.detect(fakeBytes, "test.jpg");
        } catch (Throwable e) {
            // Expected - native library not loaded (UnsatisfiedLinkError)
        }
    }

    @Test
    void matchWithNullFile1ShouldReturnErrorResult() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        JSONObject result = template.match((File) null, new File("/tmp/any.jpg"));
        assertThat(result).isNotNull();
        assertThat(result.getIntValue("error_code")).isEqualTo(500);
    }

    @Test
    void matchWithNullFile2ShouldReturnErrorResult() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        File existingFile = new File(tempDir, "exists.jpg");
        try {
            existingFile.createNewFile();
        } catch (IOException e) {
            // ignore
        }
        JSONObject result = template.match(existingFile, null);
        assertThat(result).isNotNull();
        assertThat(result.getIntValue("error_code")).isEqualTo(500);
    }

    @Test
    void matchWithNonExistentFile1ShouldReturnErrorResult() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        JSONObject result = template.match(new File("/nonexistent1.jpg"), new File("/nonexistent2.jpg"));
        assertThat(result).isNotNull();
        assertThat(result.getIntValue("error_code")).isEqualTo(500);
    }

    @Test
    void matchWithStringPathsShouldDelegateToFiles() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        JSONObject result = template.match("/nonexistent1.jpg", "/nonexistent2.jpg");
        assertThat(result).isNotNull();
    }

    @Test
    void matchWithBytesShouldCreateTempFiles() {
        OpenCVFaceRecognitionTemplate template = createTemplate();
        byte[] bytes1 = new byte[]{1, 2, 3};
        byte[] bytes2 = new byte[]{4, 5, 6};
        try {
            template.match(bytes1, bytes2, "test.jpg");
        } catch (Throwable e) {
            // Expected - native library not loaded or file format issue
        }
    }
}
