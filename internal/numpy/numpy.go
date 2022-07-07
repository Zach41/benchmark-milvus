package numpy

import (
	"fmt"
	"io"
	"os"
	"reflect"

	"github.com/sbinet/npyio/npy"
)

type DataType string

const (
	Bool    DataType = "<b1"
	UINT8   DataType = "<u1"
	UINT16  DataType = "<u2"
	UINT32  DataType = "<u4"
	UINT64  DataType = "<u8"
	INT8    DataType = "<i1"
	INT16   DataType = "<i2"
	INT32   DataType = "<i4"
	INT64   DataType = "<i8"
	FLOAT32 DataType = "<f4"
	FLOAT64 DataType = "<f8"
	UNKNOWN DataType = "<!"
)

type NumpyObject struct {
	nread   int
	cap     int
	opened  bool
	fhandle *os.File
	reader  *npy.Reader
}

func Open(fname string) (*NumpyObject, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	npyRd, err := npy.NewReader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	var cap int
	if len(npyRd.Header.Descr.Shape) <= 0 {
		cap = 0
	} else {
		cap = 1
		for i := 0; i < len(npyRd.Header.Descr.Shape); i++ {
			cap *= npyRd.Header.Descr.Shape[i]
		}
	}
	return &NumpyObject{
		nread:   0,
		cap:     cap,
		opened:  true,
		fhandle: f,
		reader:  npyRd,
	}, nil
}

func (h *NumpyObject) Close() error {
	h.opened = false
	return h.fhandle.Close()
}

func (h *NumpyObject) MetaInfo() (dims []int, dataType DataType, err error) {
	if !h.opened {
		return nil, UNKNOWN, fmt.Errorf("object closed")
	}
	return h.reader.Header.Descr.Shape, DataType(h.reader.Header.Descr.Type), nil
}

func (h *NumpyObject) Read(bufptr interface{}) (int, error) {
	if !h.opened {
		return 0, fmt.Errorf("object closed")
	}
	rv := reflect.ValueOf(bufptr)
	if !rv.IsValid() || rv.Kind() != reflect.Ptr || rv.Elem().Kind() != reflect.Slice {
		return 0, fmt.Errorf("invaid data buffer: %v", bufptr)
	}
	if rv.Elem().Len() <= 0 {
		return 0, fmt.Errorf("invalid empty buffer")
	}
	if h.nread >= h.cap {
		return 0, io.EOF
	}
	if err := h.reader.Read(bufptr); err != nil {
		return 0, err
	}
	nRead := rv.Elem().Len()
	if nRead > h.cap-h.nread {
		nRead = h.cap - h.nread
	}
	h.nread += nRead
	return nRead, nil
}

func (h *NumpyObject) ReadBoolMatrix() ([][]bool, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != Bool {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]bool, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]bool, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadInt8Matrix() ([][]int8, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != INT8 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]int8, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]int8, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadUInt8Matrix() ([][]uint8, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != UINT8 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]uint8, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]uint8, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadInt16Matrix() ([][]int16, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != INT16 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]int16, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]int16, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadUInt16Matrix() ([][]uint16, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != UINT16 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]uint16, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]uint16, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadInt32Matrix() ([][]int32, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != INT32 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]int32, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]int32, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadUInt32Matrix() ([][]uint32, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != UINT32 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]uint32, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]uint32, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadInt64Matrix() ([][]int64, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != INT64 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]int64, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]int64, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}

func (h *NumpyObject) ReadUInt64Matrix() ([][]uint64, error) {
	if !h.opened {
		return nil, fmt.Errorf("object closed")
	}
	dims, dtype, err := h.MetaInfo()
	if err != nil {
		return nil, err
	}
	if dtype != UINT64 {
		return nil, fmt.Errorf("type mismatch")
	}
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	buf := make([]uint64, dims[0]*dims[1])
	if _, err := h.Read(&buf); err != nil {
		return nil, err
	}
	ret := make([][]uint64, dims[0])
	offset := 0
	for i := 0; i < dims[0]; i++ {
		ret[i] = buf[offset : offset+dims[1]]
		offset += dims[1]
	}
	return ret, nil
}
