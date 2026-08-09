package org.bytedeco.opencv.spring.boot.nd4j.store;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Tests for {@link INDArrayInfo}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class INDArrayInfoTest {

    @Test
    void shouldCreateEmptyInfo() {
        INDArrayInfo info = new INDArrayInfo();
        assertThat(info.getGroup()).isNull();
        assertThat(info.getMemberId()).isNull();
        assertThat(info.getNdarray()).isNull();
    }

    @Test
    void shouldSetAndGetGroup() {
        INDArrayInfo info = new INDArrayInfo();
        info.setGroup("testGroup");
        assertThat(info.getGroup()).isEqualTo("testGroup");
    }

    @Test
    void shouldSetAndGetMemberId() {
        INDArrayInfo info = new INDArrayInfo();
        info.setMemberId("member1");
        assertThat(info.getMemberId()).isEqualTo("member1");
    }

    @Test
    void shouldSetAndGetNdarray() {
        INDArrayInfo info = new INDArrayInfo();
        INDArray array = Nd4j.create(new float[]{1.0f, 2.0f, 3.0f});
        info.setNdarray(array);
        assertThat(info.getNdarray()).isEqualTo(array);
    }

    @Test
    void shouldImplementEqualsAndHashCode() {
        INDArrayInfo info1 = new INDArrayInfo();
        info1.setGroup("g1");
        info1.setMemberId("m1");

        INDArrayInfo info2 = new INDArrayInfo();
        info2.setGroup("g1");
        info2.setMemberId("m1");

        assertThat(info1).isEqualTo(info2);
        assertThat(info1.hashCode()).isEqualTo(info2.hashCode());
    }

    @Test
    void shouldImplementToString() {
        INDArrayInfo info = new INDArrayInfo();
        info.setGroup("g1");
        assertThat(info.toString()).contains("g1");
    }
}
