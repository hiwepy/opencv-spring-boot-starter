package org.bytedeco.opencv.spring.boot.nd4j.store;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Tests for {@link INDArrayLocalCacheStoreProvider}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class INDArrayLocalCacheStoreProviderTest {

    @Test
    void shouldStoreAndGetArray() {
        INDArrayLocalCacheStoreProvider provider = new INDArrayLocalCacheStoreProvider();
        INDArray array = Nd4j.create(new float[]{1.0f, 2.0f, 3.0f});
        provider.store("group1", "member1", array);

        INDArray result = provider.get("group1", "member1");
        assertThat(result).isNotNull();
        assertThat(result).isEqualTo(array);
    }

    @Test
    void shouldReturnNullForNonExistentKey() {
        INDArrayLocalCacheStoreProvider provider = new INDArrayLocalCacheStoreProvider();
        INDArray result = provider.get("nonexistent", "member");
        assertThat(result).isNull();
    }

    @Test
    void shouldOverwriteExistingEntry() {
        INDArrayLocalCacheStoreProvider provider = new INDArrayLocalCacheStoreProvider();
        INDArray array1 = Nd4j.create(new float[]{1.0f});
        INDArray array2 = Nd4j.create(new float[]{2.0f});

        provider.store("group1", "member1", array1);
        provider.store("group1", "member1", array2);

        INDArray result = provider.get("group1", "member1");
        assertThat(result).isEqualTo(array2);
    }

    @Test
    void shouldStoreMultipleEntries() {
        INDArrayLocalCacheStoreProvider provider = new INDArrayLocalCacheStoreProvider();
        INDArray array1 = Nd4j.create(new float[]{1.0f});
        INDArray array2 = Nd4j.create(new float[]{2.0f});

        provider.store("g1", "m1", array1);
        provider.store("g2", "m2", array2);

        assertThat(provider.get("g1", "m1")).isEqualTo(array1);
        assertThat(provider.get("g2", "m2")).isEqualTo(array2);
    }
}
