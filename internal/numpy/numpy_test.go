package numpy

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNumpy_normal(t *testing.T) {
	obj, err := Open("test.npy")
	defer obj.Close()
	assert.Nil(t, err)

	dims, dataType, err := obj.MetaInfo()
	assert.Nil(t, err)
	assert.Equal(t, []int{250, 40}, dims)
	assert.Equal(t, INT64, dataType)

	data, err := obj.ReadInt64Matrix()
	assert.Nil(t, err)

	expected := make([][]int64, 0, 250*40)
	for i := 0; i < 250; i++ {
		row := make([]int64, 0, 40)
		for j := 0; j < 40; j++ {
			row = append(row, int64(i*40+j))
		}
		expected = append(expected, row)
	}
	assert.Equal(t, expected, data)

	_, err = obj.ReadInt64Matrix()
	assert.Error(t, err)
}

func TestNumpy_closed(t *testing.T) {
	obj, err := Open("test.npy")
	assert.Nil(t, err)
	assert.Nil(t, obj.Close())

	_, err = obj.ReadInt64Matrix()
	assert.Error(t, err)
	_, _, err = obj.MetaInfo()
	assert.Error(t, err)
}

func TestNumpy_partial_read(t *testing.T) {
	obj, err := Open("test.npy")
	assert.Nil(t, err)
	defer obj.Close()

	buf0 := make([]int64, 250*20)
	nRead, err := obj.Read(&buf0)
	assert.Nil(t, err)
	assert.Equal(t, 250*20, nRead)

	buf1 := make([]int64, 250*20)
	nRead, err = obj.Read(&buf1)
	assert.Nil(t, err)
	assert.Equal(t, 250*20, nRead)

	buf2 := make([]int64, 1)
	_, err = obj.Read(&buf2)
	assert.Error(t, err)

	expected := make([]int64, 250*40)
	for i := 0; i < len(expected); i++ {
		expected[i] = int64(i)
	}
	assert.Equal(t, expected, append(buf0, buf1...))
}

func TestNumpy_float(t *testing.T) {
	obj, err := Open("test_float.npy")
	defer obj.Close()
	assert.Nil(t, err)

	dims, dataType, err := obj.MetaInfo()
	assert.Nil(t, err)
	assert.Equal(t, []int{250, 40}, dims)
	assert.Equal(t, FLOAT64, dataType)

	data, err := obj.ReadFloat64Matrix()
	assert.Nil(t, err)

	expected := make([][]float64, 0, 250*40)
	for i := 0; i < 250; i++ {
		row := make([]float64, 0, 40)
		for j := 0; j < 40; j++ {
			row = append(row, float64(i*40+j))
		}
		expected = append(expected, row)
	}
	assert.Equal(t, expected, data)

	_, err = obj.ReadInt64Matrix()
	assert.Error(t, err)
}
