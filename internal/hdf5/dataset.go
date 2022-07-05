package hdf5

import "fmt"

type Dataset struct {
	dims []uint
	data interface{}
}

func (d *Dataset) ReadInt() (dims []uint, data []int32, err error) {
	intdata, ok := d.data.([]int32)
	if !ok {
		return nil, nil, fmt.Errorf("invalid int dataset")
	}
	return d.dims, intdata, nil
}

func (d *Dataset) ReadFloats() (dims []uint, data []float32, err error) {
	floatdata, ok := d.data.([]float32)
	if !ok {
		return nil, nil, fmt.Errorf("invalid float dataset")
	}
	return d.dims, floatdata, nil
}

func (d *Dataset) ReadIntMatrix() ([][]int32, error) {
	dims, rawdata, err := d.ReadInt()
	if err != nil {
		return nil, err
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid matrix data, dimensions: %v", dims)
	}
	data := make([][]int32, dims[0])
	var offset, dim uint = 0, dims[1]
	var idx uint
	for idx = 0; idx < dims[0]; idx++ {
		data[idx] = make([]int32, dims[1])
		copy(data[idx], rawdata[offset:offset+dim])
		offset += dim
	}
	return data, nil
}

func (d *Dataset) ReadFloatMatrix() ([][]float32, error) {
	dims, rawdata, err := d.ReadFloats()
	if err != nil {
		return nil, err
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid matrix data, dimensions: %v", dims)
	}
	data := make([][]float32, dims[0])
	var offset, dim uint = 0, dims[1]
	var idx uint
	for idx = 0; idx < dims[0]; idx++ {
		data[idx] = make([]float32, dims[1])
		copy(data[idx], rawdata[offset:offset+dim])
		offset += dim
	}
	return data, nil
}
