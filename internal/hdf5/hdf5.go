package hdf5

import (
	"fmt"
	"path"
	"strings"

	"gonum.org/v1/hdf5"
)

type ObjectType int

const (
	H5File ObjectType = iota
	H5Group
	H5Dataset
)

type H5Object struct {
	opened bool
	name   string
	handle interface{}
	data   *Dataset
	childs []*H5Object
}

func Open(fname string) (obj *H5Object, err error) {
	defer func() {
		if err != nil && obj != nil {
			obj.Close()
		}
	}()
	if !(hdf5.IsHDF5(fname)) {
		return nil, fmt.Errorf("invalid hdf5 file")
	}
	f, err := hdf5.OpenFile(fname, hdf5.F_ACC_RDONLY)
	if err != nil {
		return nil, err
	}
	obj = &H5Object{
		opened: false,
		name:   "",
		handle: f,
		data:   nil,
		childs: nil,
	}
	if err := obj.readCommonFG(); err != nil {
		return nil, err
	}
	obj.opened = true
	return obj, nil
}

func (h *H5Object) Close() error {
	for _, obj := range h.childs {
		if obj == nil {
			continue
		}
		if err := obj.Close(); err != nil {
			return err
		}
	}
	h.opened = false
	switch f := h.handle.(type) {
	case *hdf5.Group:
		return f.Close()
	case *hdf5.Dataset:
		return f.Close()
	case *hdf5.File:
		return f.Close()
	}
	return nil
}

func (h *H5Object) readCommonFG() (err error) {
	var fg *hdf5.CommonFG
	switch f := h.handle.(type) {
	case *hdf5.File:
		fg = &f.CommonFG
	case *hdf5.Group:
		fg = &f.CommonFG
	default:
		return nil
	}

	objCnt, err := fg.NumObjects()
	if err != nil {
		return err
	}
	h.childs = make([]*H5Object, 0, objCnt)
	var idx uint
	for idx = 0; idx < objCnt; idx++ {
		objTy, err := fg.ObjectTypeByIndex(idx)
		if err != nil {
			return err
		}
		objName, err := fg.ObjectNameByIndex(idx)
		if err != nil {
			return err
		}
		switch objTy {
		case hdf5.H5G_GROUP:
			group, err := fg.OpenGroup(objName)
			if err != nil {
				return err
			}
			child := &H5Object{
				name:   objName,
				handle: group,
				data:   nil,
				childs: nil,
			}
			if err := child.readCommonFG(); err != nil {
				return err
			}
			h.childs = append(h.childs, child)
		case hdf5.H5G_DATASET:
			dataset, err := fg.OpenDataset(objName)
			if err != nil {
				return err
			}
			child := &H5Object{
				name:   objName,
				handle: dataset,
				data:   nil,
				childs: nil,
			}
			if err := child.readDataset(); err != nil {
				return err
			}
			h.childs = append(h.childs, child)
		default:
			// TODO: read objs
		}
	}
	return nil
}

func (h *H5Object) readDataset() error {
	f, ok := h.handle.(*hdf5.Dataset)
	if !ok {
		return nil
	}
	dataspace := f.Space()
	defer dataspace.Close()

	dims, _, err := dataspace.SimpleExtentDims()
	if err != nil {
		return err
	}

	dataType, err := f.Datatype()
	if err != nil {
		return err
	}
	defer dataType.Close()
	var bufsz uint = 1
	for i := 0; i < len(dims); i++ {
		bufsz *= dims[i]
	}

	switch dataType.Class() {
	case hdf5.T_FLOAT:
		arr := make([]float32, bufsz)
		if err := f.Read(&arr[0]); err != nil {
			return err
		}
		h.data = &Dataset{
			dims: dims,
			data: arr,
		}
	case hdf5.T_INTEGER:
		arr := make([]int, bufsz)
		if err := f.Read(&arr[0]); err != nil {
			return err
		}
		h.data = &Dataset{
			dims: dims,
			data: arr,
		}
	default:
		return fmt.Errorf("unspported dataset type")
	}
	return nil
}

func (h *H5Object) Read(dpath string) (*Dataset, error) {
	if !h.opened {
		return nil, fmt.Errorf("object not open")
	}
	paths := strings.Split(dpath, "/")
	if paths[0] != h.name {
		return nil, fmt.Errorf("not found")
	}
	d := h.search(paths[1:])
	if d == nil {
		return nil, fmt.Errorf("not found")
	}
	return d, nil
}

func (h *H5Object) search(dpaths []string) *Dataset {
	if len(dpaths) == 0 {
		return h.data
	}
	for i := 0; i < len(h.childs); i++ {
		if h.childs[i] != nil && h.childs[i].name == dpaths[0] {
			return h.childs[i].search(dpaths[1:])
		}
	}
	return nil
}

func (h *H5Object) AllDataPaths() []string {
	if !h.opened {
		return nil
	}
	return h.dpaths("/")
}

func (h *H5Object) dpaths(prefix string) []string {
	switch h.handle.(type) {
	case *hdf5.Dataset:
		return []string{path.Join(prefix, h.name)}
	case *hdf5.Group, *hdf5.File:
		prefix = path.Join(prefix, h.name)
		dpaths := make([]string, 0, len(h.childs))
		for _, child := range h.childs {
			if child != nil {
				subPaths := child.dpaths(prefix)
				dpaths = append(dpaths, subPaths...)
			}
		}
		return dpaths
	}
	return nil
}
