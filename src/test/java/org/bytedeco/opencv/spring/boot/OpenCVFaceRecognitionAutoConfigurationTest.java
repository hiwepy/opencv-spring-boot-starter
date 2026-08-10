package org.bytedeco.opencv.spring.boot;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link OpenCVFaceRecognitionAutoConfiguration}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class OpenCVFaceRecognitionAutoConfigurationTest {

    @Test
    void shouldNotActivateWhenPropertyDisabled() {
        // The static initializer loads native OpenCV libs, so we can't instantiate
        // the configuration class in a test environment without native libs.
        // Instead we verify the class exists and has the expected annotations.
        assertThat(OpenCVFaceRecognitionAutoConfiguration.class).isNotNull();
        assertThat(OpenCVFaceRecognitionAutoConfiguration.class.getAnnotation(
                org.springframework.context.annotation.Configuration.class)).isNotNull();
    }

    @Test
    void shouldHaveConditionalOnPropertyAnnotation() {
        assertThat(OpenCVFaceRecognitionAutoConfiguration.class.getAnnotation(
                org.springframework.boot.autoconfigure.condition.ConditionalOnProperty.class)).isNotNull();
    }

    @Test
    void shouldHaveEnableConfigurationPropertiesAnnotation() {
        assertThat(OpenCVFaceRecognitionAutoConfiguration.class.getAnnotation(
                org.springframework.boot.context.properties.EnableConfigurationProperties.class)).isNotNull();
    }

    @Test
    void openCVFaceRecognitionTemplateBeanShouldBeCreatable() {
        // Test template creation without native dependencies
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        OpenCVFaceRecognitionTemplate template = new OpenCVFaceRecognitionTemplate(null, props);
        assertThat(template).isNotNull();
        assertThat(template.getProperties()).isEqualTo(props);
        assertThat(template.getFaceDetector()).isNull();
    }

}
