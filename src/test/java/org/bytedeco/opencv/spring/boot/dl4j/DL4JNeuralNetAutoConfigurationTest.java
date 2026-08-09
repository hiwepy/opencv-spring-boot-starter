package org.bytedeco.opencv.spring.boot.dl4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.bytedeco.opencv.spring.boot.OpenCVFaceRecognitionProperties;
import org.bytedeco.opencv.spring.boot.OpenCVFaceRecognitionTemplate;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link DL4JNeuralNetAutoConfiguration}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class DL4JNeuralNetAutoConfigurationTest {

    @Test
    void shouldNotActivateWhenPropertyDisabled() {
        // The configuration class has @ConditionalOnProperty, so it won't activate
        // when the property is not set or set to false.
        assertThat(DL4JNeuralNetAutoConfiguration.class).isNotNull();
        assertThat(DL4JNeuralNetAutoConfiguration.class.getAnnotation(
                org.springframework.context.annotation.Configuration.class)).isNotNull();
    }

    @Test
    void shouldHaveConditionalOnPropertyAnnotation() {
        assertThat(DL4JNeuralNetAutoConfiguration.class.getAnnotation(
                org.springframework.boot.autoconfigure.condition.ConditionalOnProperty.class)).isNotNull();
    }

    @Test
    void shouldHaveEnableConfigurationPropertiesAnnotation() {
        assertThat(DL4JNeuralNetAutoConfiguration.class.getAnnotation(
                org.springframework.boot.context.properties.EnableConfigurationProperties.class)).isNotNull();
    }

    @Test
    void faceNetSmallV2ModelBeanShouldBeCreatable() {
        DL4JNeuralNetAutoConfiguration config = new DL4JNeuralNetAutoConfiguration();
        FaceNetSmallV2Model model = config.faceNetSmallV2Model();
        assertThat(model).isNotNull();
    }

    @Test
    void openCVFaceRecognitionTemplateBeanShouldBeCreatable() {
        DL4JNeuralNetAutoConfiguration config = new DL4JNeuralNetAutoConfiguration();
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        OpenCVFaceRecognitionTemplate template = config.openCVFaceRecognitionTemplate(null, props);
        assertThat(template).isNotNull();
    }

}
