package org.bytedeco.opencv.spring.boot.image;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for {@link ImageFactory}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class ImageFactoryTest {

    @TempDir
    File tempDir;

    private BufferedImage createTestImage(int width, int height, int type) {
        BufferedImage image = new BufferedImage(width, height, type);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, 0xFF0000FF); // blue pixel
            }
        }
        return image;
    }

    @Test
    void getRGBDataShouldReturnNullForNullFile() {
        assertThat(ImageFactory.getRGBData((File) null)).isNull();
    }

    @Test
    void getGrayDataShouldReturnNullForNullFile() {
        assertThat(ImageFactory.getGrayData((File) null)).isNull();
    }

    @Test
    void getRGBDataShouldReturnNullForNullBytes() {
        assertThat(ImageFactory.getRGBData((byte[]) null)).isNull();
    }

    @Test
    void getGrayDataShouldReturnNullForNullBytes() {
        assertThat(ImageFactory.getGrayData((byte[]) null)).isNull();
    }

    @Test
    void getRGBDataShouldReturnNullForNullInputStream() {
        assertThat(ImageFactory.getRGBData((InputStream) null)).isNull();
    }

    @Test
    void getGrayDataShouldReturnNullForNullInputStream() {
        assertThat(ImageFactory.getGrayData((InputStream) null)).isNull();
    }

    @Test
    void getRGBDataFromFileShouldReturnImageInfo() throws Exception {
        File imageFile = new File(tempDir, "test_rgb.png");
        BufferedImage image = createTestImage(64, 64, BufferedImage.TYPE_3BYTE_BGR);
        ImageIO.write(image, "png", imageFile);

        ImageInfo info = ImageFactory.getRGBData(imageFile);
        assertThat(info).isNotNull();
        assertThat(info.getWidth()).isEqualTo(64);
        assertThat(info.getHeight()).isEqualTo(64);
        assertThat(info.getImageFormat()).isEqualTo(ImageFormat.CP_PAF_BGR24);
        assertThat(info.getImageData()).isNotNull();
        assertThat(info.getImageData().length).isGreaterThan(0);
    }

    @Test
    void getGrayDataFromFileShouldReturnImageInfo() throws Exception {
        File imageFile = new File(tempDir, "test_gray.png");
        BufferedImage image = createTestImage(64, 64, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(image, "png", imageFile);

        ImageInfo info = ImageFactory.getGrayData(imageFile);
        assertThat(info).isNotNull();
        assertThat(info.getWidth()).isEqualTo(64);
        assertThat(info.getHeight()).isEqualTo(64);
        assertThat(info.getImageFormat()).isEqualTo(ImageFormat.CP_PAF_GRAY);
        assertThat(info.getImageData()).isNotNull();
    }

    @Test
    void getRGBDataFromBytesShouldReturnImageInfo() throws Exception {
        BufferedImage image = createTestImage(32, 32, BufferedImage.TYPE_3BYTE_BGR);
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        byte[] imageBytes = baos.toByteArray();

        ImageInfo info = ImageFactory.getRGBData(imageBytes);
        assertThat(info).isNotNull();
        assertThat(info.getWidth()).isEqualTo(32);
        assertThat(info.getHeight()).isEqualTo(32);
    }

    @Test
    void getGrayDataFromBytesShouldReturnImageInfo() throws Exception {
        BufferedImage image = createTestImage(32, 32, BufferedImage.TYPE_INT_RGB);
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        byte[] imageBytes = baos.toByteArray();

        ImageInfo info = ImageFactory.getGrayData(imageBytes);
        assertThat(info).isNotNull();
        assertThat(info.getWidth()).isEqualTo(32);
        assertThat(info.getHeight()).isEqualTo(32);
    }

    @Test
    void getRGBDataFromInputStreamShouldReturnImageInfo() throws Exception {
        BufferedImage image = createTestImage(16, 16, BufferedImage.TYPE_3BYTE_BGR);
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        byte[] imageBytes = baos.toByteArray();

        ImageInfo info = ImageFactory.getRGBData(new ByteArrayInputStream(imageBytes));
        assertThat(info).isNotNull();
        assertThat(info.getWidth()).isEqualTo(16);
        assertThat(info.getHeight()).isEqualTo(16);
    }

    @Test
    void getGrayDataFromInputStreamShouldReturnImageInfo() throws Exception {
        BufferedImage image = createTestImage(16, 16, BufferedImage.TYPE_INT_RGB);
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        byte[] imageBytes = baos.toByteArray();

        ImageInfo info = ImageFactory.getGrayData(new ByteArrayInputStream(imageBytes));
        assertThat(info).isNotNull();
        assertThat(info.getWidth()).isEqualTo(16);
        assertThat(info.getHeight()).isEqualTo(16);
    }

    @Test
    void getRGBDataFromInputStreamShouldReturnNullForInvalidData() {
        byte[] garbage = new byte[]{0, 1, 2, 3};
        ImageInfo info = ImageFactory.getRGBData(new ByteArrayInputStream(garbage));
        assertThat(info).isNull();
    }

    @Test
    void getGrayDataFromInputStreamShouldReturnNullForInvalidData() {
        byte[] garbage = new byte[]{0, 1, 2, 3};
        ImageInfo info = ImageFactory.getGrayData(new ByteArrayInputStream(garbage));
        assertThat(info).isNull();
    }

    @Test
    void bufferedImage2ImageInfoShouldHandleNonBGRType() {
        BufferedImage image = createTestImage(16, 16, BufferedImage.TYPE_INT_ARGB);
        ImageInfo info = ImageFactory.bufferedImage2ImageInfo(image);
        assertThat(info).isNotNull();
        assertThat(info.getImageFormat()).isEqualTo(ImageFormat.CP_PAF_BGR24);
        assertThat(info.getImageData()).isNotNull();
    }

    @Test
    void getBestRectShouldReturnNullForNullInput() {
        assertThat(ImageFactory.getBestRect(100, 100, null)).isNull();
    }

    @Test
    void getBestRectShouldExpandRect() {
        Rect src = new Rect(40, 40, 60, 60);
        Rect result = ImageFactory.getBestRect(100, 100, src);
        assertThat(result).isNotNull();
        assertThat(result.getLeft()).isLessThan(40);
        assertThat(result.getTop()).isLessThan(40);
        assertThat(result.getRight()).isGreaterThan(60);
        assertThat(result.getBottom()).isGreaterThan(60);
    }

    @Test
    void getBestRectShouldHandleOverflow() {
        Rect src = new Rect(0, 0, 10, 10);
        Rect result = ImageFactory.getBestRect(50, 50, src);
        assertThat(result).isNotNull();
        assertThat(result.getLeft()).isGreaterThanOrEqualTo(0);
        assertThat(result.getTop()).isGreaterThanOrEqualTo(0);
    }

    @Test
    void getBestRectShouldHandleRectAtBoundary() {
        Rect src = new Rect(5, 5, 45, 45);
        Rect result = ImageFactory.getBestRect(50, 50, src);
        assertThat(result).isNotNull();
    }

    @Test
    void getBestRectShouldHandleRectWithNegativeOverflow() {
        Rect src = new Rect(-5, 10, 30, 40);
        Rect result = ImageFactory.getBestRect(50, 50, src);
        assertThat(result).isNotNull();
        assertThat(result.getLeft()).isGreaterThanOrEqualTo(0);
    }

    @Test
    void getBestRectShouldHandleRectExceedingWidth() {
        Rect src = new Rect(10, 10, 60, 40);
        Rect result = ImageFactory.getBestRect(50, 50, src);
        assertThat(result).isNotNull();
        assertThat(result.getRight()).isLessThanOrEqualTo(50);
    }

    @Test
    void getBestRectShouldHandleRectExceedingHeight() {
        Rect src = new Rect(10, 10, 40, 60);
        Rect result = ImageFactory.getBestRect(50, 50, src);
        assertThat(result).isNotNull();
        assertThat(result.getBottom()).isLessThanOrEqualTo(50);
    }

    @Test
    void getBestRectShouldHandleSmallRectInLargeImage() {
        Rect src = new Rect(20, 20, 30, 30);
        Rect result = ImageFactory.getBestRect(200, 200, src);
        assertThat(result).isNotNull();
        assertThat(result.getLeft()).isLessThan(20);
        assertThat(result.getTop()).isLessThan(20);
        assertThat(result.getRight()).isGreaterThan(30);
        assertThat(result.getBottom()).isGreaterThan(30);
    }

    @Test
    void getGrayDataFromFileShouldReturnNullForNonExistentFile() {
        ImageInfo info = ImageFactory.getGrayData(new File("/nonexistent/path/image.png"));
        assertThat(info).isNull();
    }

    @Test
    void getRGBDataFromFileShouldReturnNullForNonExistentFile() {
        ImageInfo info = ImageFactory.getRGBData(new File("/nonexistent/path/image.png"));
        assertThat(info).isNull();
    }

    @Test
    void bufferedImage2GrayImageInfoShouldHandleGrayImage() {
        BufferedImage image = createTestImage(16, 16, BufferedImage.TYPE_BYTE_GRAY);
        ImageInfo info = ImageFactory.bufferedImage2GrayImageInfo(image);
        assertThat(info).isNotNull();
        assertThat(info.getImageFormat()).isEqualTo(ImageFormat.CP_PAF_GRAY);
    }

}
