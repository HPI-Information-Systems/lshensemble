package main

import ( //"github.com/ekzhu/lshensemble"
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/ekzhu/lshensemble"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strings"
	"time"
)

type Domain struct {
	PageID json.Number `json:"pageID"''`
	TableHID string `json:"TableHID"''`
	TID int `json:"tableID"''`
	ColID int `json:"colID"''`
	Values []string `json:"values"''`
}

func main() {

	programStart := time.Now()
	files, err := ioutil.ReadDir(os.Args[1])
	trainFiles, _ := ioutil.ReadDir(os.Args[2])

	queryFiles := filterByName(files,"query")
	indexFiles := filterByName(files,"index")

	//
	domainsToIndex, keys, _ := readAllDomains(indexFiles,trainFiles)

	// initializing the domain records to hold the MinHash signatures
	domainRecords := make([]*lshensemble.DomainRecord, len(domainsToIndex))

	// set the minhash seed
	var seed int64 = 42

	// set the number of hash functions
	numHash := 256

	println("Done Reading files, beginning to processing domainsToIndex")
	// create the domain records with the signatures
	for i := range domainsToIndex {
		mh := lshensemble.NewMinhash(seed, numHash)
		for v := range domainsToIndex[i] {
			mh.Push([]byte(v))
		}
		domainRecords[i] = &lshensemble.DomainRecord{
			Key:       keys[i],
			Size:      len(domainsToIndex[i]),
			Signature: mh.Signature()}
	}
	println("Done processing files")

	//SECOND STEP:

	// Set the number of partitions
	numPart := 8

	// Set the maximum value for the MinHash LSH parameter K
	// (number of hash functions per band).
	maxK := 4

	//SECOND AND HALF STEP: sort
	sort.Sort(lshensemble.BySize(domainRecords))

	println("Done Sorting")

	// Create index using optimal partitioning
	// You can also use BootstrapLshEnsemblePlusOptimal for better accuracy
	index, err := lshensemble.BootstrapLshEnsembleOptimal(numPart, numHash, maxK,
		func() <-chan *lshensemble.DomainRecord {
			return lshensemble.Recs2Chan(domainRecords);
		})
	if err != nil {
		panic(err)
	}
	println("Done Indexing")
	indexingTime := time.Since(programStart)
	println(fmt.Sprintf("Build Index in %v hours",indexingTime.Hours()))
	curQueryingStart := time.Now()
	if err != nil {
		fmt.Println(err)
		return
	}
	//file pointer for train files:
	var thresholds [] float64
	thresholds = append(thresholds,0.8)
	thresholds = append(thresholds,0.9)
	thresholds = append(thresholds,1.0)
	var fileCount = 0
	for _, f := range trainFiles {
		if (fileCount%10 == 0) {
			println(fmt.Sprintf("Read %v of %v files", fileCount, len(queryFiles)))
		}
		var fname= os.Args[2] + "/" + f.Name()
		println("processing file "+f.Name())
		var curFileStart = time.Now()
		var curOutFile,_ = os.Create(os.Args[3] + f.Name() + "_features.csv")
		writeFileHeader(curOutFile,thresholds)
		processQueryFile(fname, f, seed, numHash, curQueryingStart, index, thresholds,curOutFile)
		println(fmt.Sprintf("Total Runtime for file %v [h]: %v",f.Name(),time.Since(curFileStart).Hours()))
		curOutFile.Close()
		fileCount++
	}
	println("Done with querying train Files")
	for _, f := range queryFiles {
		if (fileCount%10 == 0) {
			println(fmt.Sprintf("Read %v of %v files", fileCount, len(queryFiles)))
		}
		var fname= os.Args[1] + "/" + f.Name()
		println("processing file "+f.Name())
		var curFileStart = time.Now()
		var curOutFile,_ = os.Create(os.Args[3] + f.Name() + "_features.csv")
		writeFileHeader(curOutFile,thresholds)
		processQueryFile(fname, f, seed, numHash, curQueryingStart, index, thresholds, curOutFile)
		println(fmt.Sprintf("Total Runtime for file %v [h]: %v",f.Name(),time.Since(curFileStart).Hours()))
		curOutFile.Close()
		fileCount++
	}
	println(fmt.Sprintf("Total Runtime for entire Feature Extraction [h]: %v",time.Since(programStart).Hours()))
}

func processQueryFile(fname string, inputFile os.FileInfo, seed int64, numHash int, curQueryingStart time.Time, index *lshensemble.LshEnsemble, thresholds []float64, outFile *os.File) {
	var count = 0
	var queryKeys, queryDomains = readDomains(fname,false)
	for i := range queryDomains {
		mh := lshensemble.NewMinhash(seed, numHash)
		for v := range queryDomains[i] {
			mh.Push([]byte(v))
		}
		curRecord := &lshensemble.DomainRecord{
			Key:       queryKeys[i],
			Size:      len(queryDomains[i]),
			Signature: mh.Signature()}
		count++
		if (count%1000 == 0) {
			batchTime := time.Since(curQueryingStart)
			println(fmt.Sprintf("In file %v it took %v minutes to query %v domains out of %v (%v%%)", inputFile.Name(), batchTime.Minutes(), count, len(queryKeys), 100.0*float64(count)/float64(len(queryKeys))))
			curQueryingStart = time.Now()
		}
		// query in the domain records:
		querySig := curRecord.Signature
		querySize := curRecord.Size
		var queryKey = curRecord.Key
		var queryKeyString = fmt.Sprintf("%v", queryKey)
		var matchCounts [] int
		for i := range thresholds {
			done := make(chan struct{})
			defer close(done) // Important!!
			// set the containment startingThreshold
			// get the keys of the candidate domainsToIndex (may contain false positives)
			// through a channel with option to cancel early.
			results := index.Query(querySig, querySize, thresholds[i], done)
			resCount := 0
			for range results {
				resCount++
			}
			matchCounts = append(matchCounts, resCount)
		}
		var keyVals = strings.Split(queryKeyString, "||||")
		var pageID = keyVals[0]
		var tableHID = keyVals[1]
		var tableID = keyVals[2]
		var column = keyVals[3]
		//result writing new:
		outFile.WriteString(fmt.Sprintf("%v,%v,%v,%v,", pageID, tableHID, tableID, column))
		for i := range thresholds {
			if (i == len(thresholds)-1) {
				outFile.WriteString(fmt.Sprintf("%v\n", matchCounts[i]))
			} else {
				outFile.WriteString(fmt.Sprintf("%v,", matchCounts[i]))
			}
		}
	}
}

func filterByName(infos []os.FileInfo, s string) []os.FileInfo {
	filtered := make([]os.FileInfo, 0, len(infos))
	for _, f := range infos{
		if(strings.Contains(f.Name(),s)){
			filtered = append(filtered,f)
		}
	}
	return filtered
}

func writeFileHeader(f *os.File, thresholds []float64) {
	f.WriteString("pageID, tableHID, tableID, column,")
	for i := range thresholds {
		if (i == len(thresholds)-1) {
			f.WriteString(fmt.Sprintf("containmentAt_%v\n", thresholds[i]))
		} else {
			f.WriteString(fmt.Sprintf("containmentAt_%v,", thresholds[i]))
		}
	}
}

func addMatchCounts(query []string, candidate []string,thresholds []float64,matchCounts []int) {
	var queryMap = toMap(query)
	var candidateMap = toMap(candidate)
	var containmentCount = 0
	for k := range queryMap {
		if _, ok := candidateMap[k]; ok { containmentCount++}
	}
	for i := range thresholds {
		if(float64(containmentCount) / float64(len(queryMap)) >= thresholds[i]){
			matchCounts[i]++
		}
	}
}

func toMap(elements []string) map[string]bool {
	elementMap := make(map[string]bool)
	for i := 0; i < len(elements); i += 1 {
		elementMap[elements[i]] = true
	}
	return elementMap
}

func readAllDomains(files []os.FileInfo,trainFiles []os.FileInfo) ([]map[string]bool, []string, map[string]bool) {
	var domains [] map[string]bool
	var keys [] string
	//files = files[1:10]
	var count = 0
	for _, f := range files {
		if (count%10 == 0) {
			println(fmt.Sprintf("Read %v of %v files", count, len(files)))
		}
		var fname= os.Args[1] + "/" + f.Name()
		var key, domain= readDomains(fname, false)
		keys = append(keys, key...)
		domains = append(domains, domain...)
		count++
	}
	var myTableKeys = make(map[string]bool)
	for _, f := range trainFiles {
		println(fmt.Sprintf("Reading %v", f))
		var fname = os.Args[2] + "/" + f.Name()
		var key,domain = readDomains(fname,true)
		for i := range key{
			myTableKeys[key[i]] = true
		}
		keys = append(keys, key...)
		domains = append(domains, domain...)
	}
	return domains,keys, myTableKeys
}

func readDomains(jsonFile string, keepSingleElem bool) ([]string,[]map[string]bool) {
	file, err := os.Open(jsonFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var keys [] string
	var domains [] map[string]bool

	var line []byte
	var errLR error
	var errJson error
	for {
		var domainJson Domain
		line, errLR = reader.ReadBytes('\n')
		if errLR != nil {
			break
		}
		errJson = json.Unmarshal(line, &domainJson)
		var m = make(map[string]bool)
		if(errJson!=nil){
			println("Error while parsing json")
			break
		}
		for j := 0; j < len(domainJson.Values); j++ {
			m[domainJson.Values[j]] = true
		}
		if(keepSingleElem || len(m)>1) {
			var domainKey = getDomainKey(domainJson)
			keys = append(keys,domainKey)
			domains = append(domains, m)
		}

	}
	return keys,domains
}

func getDomainKey(domain Domain) string {
	return fmt.Sprintf("%v||||%v||||%v||||%v", domain.PageID, domain.TableHID, domain.TID, domain.ColID)
}

