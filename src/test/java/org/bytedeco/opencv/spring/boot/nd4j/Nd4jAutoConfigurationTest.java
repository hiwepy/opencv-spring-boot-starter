package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Nd4jAutoConfiguration}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class Nd4jAutoConfigurationTest {

    @Test
    void shouldNotActivateWhenPropertyDisabled() {
        assertThat(Nd4jAutoConfiguration.class).isNotNull();
        assertThat(Nd4jAutoConfiguration.class.getAnnotation(
                org.springframework.context.annotation.Configuration.class)).isNotNull();
    }

    @Test
    void shouldHaveConditionalOnPropertyAnnotation() {
        assertThat(Nd4jAutoConfiguration.class.getAnnotation(
                org.springframework.boot.autoconfigure.condition.ConditionalOnProperty.class)).isNotNull();
    }

    @Test
    void imageLoaderWithNATIVEShouldCreateNativeLoader() {
        Nd4jAutoConfiguration config = new Nd4jAutoConfiguration();
        Nd4jProperties props = new Nd4jProperties();
        props.setType(ImageLoader.NATIVE);

        Nd4jCifarLoaderProperties cifarProps = new Nd4jCifarLoaderProperties();
        Nd4jImageLoaderProperties loaderProps = new Nd4jImageLoaderProperties();
        Nd4jLFWLoaderProperties lfwProps = new Nd4jLFWLoaderProperties();
        Nd4NativeLoaderProperties nativeProps = new Nd4NativeLoaderProperties();

        BaseImageLoader loader = config.imageLoader(props, cifarProps, loaderProps, lfwProps, nativeProps);
        assertThat(loader).isInstanceOf(NativeImageLoader.class);
    }

    @Test
    void imageLoaderWithDEFAULTShouldCreateImageLoader() {
        Nd4jAutoConfiguration config = new Nd4jAutoConfiguration();
        Nd4jProperties props = new Nd4jProperties();
        props.setType(ImageLoader.DEFAULT);

        Nd4jCifarLoaderProperties cifarProps = new Nd4jCifarLoaderProperties();
        Nd4jImageLoaderProperties loaderProps = new Nd4jImageLoaderProperties();
        loaderProps.setHeight(64L);
        loaderProps.setWidth(64L);
        loaderProps.setChannels(3L);
        Nd4jLFWLoaderProperties lfwProps = new Nd4jLFWLoaderProperties();
        Nd4NativeLoaderProperties nativeProps = new Nd4NativeLoaderProperties();

        BaseImageLoader loader = config.imageLoader(props, cifarProps, loaderProps, lfwProps, nativeProps);
        assertThat(loader).isInstanceOf(org.datavec.image.loader.ImageLoader.class);
    }

    @Test
    void imageLoaderWithLFWShouldCreateLFWLoader() {
        Nd4jAutoConfiguration config = new Nd4jAutoConfiguration();
        Nd4jProperties props = new Nd4jProperties();
        props.setType(ImageLoader.LFW);

        Nd4jCifarLoaderProperties cifarProps = new Nd4jCifarLoaderProperties();
        Nd4jImageLoaderProperties loaderProps = new Nd4jImageLoaderProperties();
        Nd4jLFWLoaderProperties lfwProps = new Nd4jLFWLoaderProperties();
        Nd4NativeLoaderProperties nativeProps = new Nd4NativeLoaderProperties();

        BaseImageLoader loader = config.imageLoader(props, cifarProps, loaderProps, lfwProps, nativeProps);
        assertThat(loader).isInstanceOf(org.datavec.image.loader.LFWLoader.class);
    }

    @Test
    void shouldHaveEnableConfigurationPropertiesAnnotation() {
        assertThat(Nd4jAutoConfiguration.class.getAnnotation(
                org.springframework.boot.context.properties.EnableConfigurationProperties.class)).isNotNull();
    }

}
