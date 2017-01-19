package lshensemble

import (
	"sort"
	"testing"
)

func Test_LshEnsemble(t *testing.T) {
	domains := [][]string{
		[]string{"a", "b", "c", "d"},
		[]string{"e", "f", "g", "h"},
		[]string{"i", "j", "k", "l"},
		[]string{"p", "o", "n", "m"},
	}
	keys := []string{
		"1",
		"2",
		"3",
		"4",
	}
	domainRecords := make([]*DomainRecord, 0)
	for i := range domains {
		mh := NewMinhash(1, 128)
		for _, v := range domains[i] {
			mh.Push([]byte(v))
		}
		domainRecords = append(domainRecords, &DomainRecord{
			Key:       keys[i],
			Size:      len(domains[i]),
			Signature: mh.Signature(),
		})
	}
	sort.Sort(BySize(domainRecords))
	index := BootstrapLshEnsemble(4, 128, 4, len(domainRecords),
		Recs2Chan(domainRecords))

	querySig := domainRecords[0].Signature
	querySize := domainRecords[0].Size
	threshold := 0.9
	var found bool
	for key := range index.Query(querySig, querySize, threshold) {
		if key == "1" {
			found = true
		}
	}
	if !found {
		t.Fatal("unable to retrieve inserted key")
	}
}