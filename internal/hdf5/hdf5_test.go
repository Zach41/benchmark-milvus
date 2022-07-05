package hdf5

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHDF5(t *testing.T) {
	h, err := Open("../../benchmark-data/glove-25-angular.hdf5")
	assert.Nil(t, err)
	defer h.Close()

	dpaths := h.AllDataPaths()

	assert.ElementsMatch(t, dpaths, []string{"/distances", "/neighbors", "/test", "/train"})

	dataset, err := h.Read("/test")
	assert.Nil(t, err)
	assert.NotNil(t, dataset)

	_, err = dataset.ReadFloatMatrix()
	assert.Nil(t, err)

	_, err = h.Read("/not_exist")
	assert.Error(t, err)
}

func TestReadClosed(t *testing.T) {
	h, err := Open("../../benchmark-data/glove-25-angular.hdf5")
	assert.Nil(t, err)
	err = h.Close()
	assert.Nil(t, err)

	dpaths := h.AllDataPaths()
	assert.Nil(t, dpaths)

	_, err = h.Read("/")
	assert.Error(t, err)
}
