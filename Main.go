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
	Id       string   //`json:"Id"''`
	version  string   //`json:"version"''` //TODO: fix this
	AttrName string   //`json:"AttrName"''`
	Values   []string `json:"values"''`
}

func main() {

	programStart := time.Now()
	files, err := ioutil.ReadDir(os.Args[1])

	queryFiles := filterByName(files, "query")
	indexFiles := filterByName(files, "index")
	if len(queryFiles) == 0 {
		queryFiles = indexFiles
	}
	//
	domainsToIndex, keys := readAllDomains(indexFiles)

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
			return lshensemble.Recs2Chan(domainRecords)
		})
	if err != nil {
		panic(err)
	}
	println("Done Indexing")
	indexingTime := time.Since(programStart)
	println(fmt.Sprintf("Build Index in %v hours", indexingTime.Hours()))
	curQueryingStart := time.Now()
	if err != nil {
		fmt.Println(err)
		return
	}
	//file pointer for train files:
	var thresholds []float64
	thresholds = append(thresholds, 0.8)
	thresholds = append(thresholds, 0.9)
	thresholds = append(thresholds, 1.0)
	var fileCount = 0
	for _, f := range queryFiles {
		if fileCount%10 == 0 {
			println(fmt.Sprintf("Read %v of %v files", fileCount, len(queryFiles)))
		}
		var fname = os.Args[1] + "/" + f.Name()
		println("processing file " + f.Name())
		var curFileStart = time.Now()
		var curOutFile, _ = os.Create(os.Args[2] + f.Name() + "_joinabilityGraph.csv")
		writeFileHeader(curOutFile, thresholds)
		processQueryFile(fname, f, seed, numHash, curQueryingStart, index, thresholds, curOutFile)
		println(fmt.Sprintf("Total Runtime for file %v [h]: %v", f.Name(), time.Since(curFileStart).Hours()))
		curOutFile.Close()
		fileCount++
	}
	println(fmt.Sprintf("Total Runtime for entire Feature Extraction [h]: %v", time.Since(programStart).Hours()))
}

func processQueryFile(fname string, inputFile os.FileInfo, seed int64, numHash int, curQueryingStart time.Time, index *lshensemble.LshEnsemble, thresholds []float64, outFile *os.File) {
	var count = 0
	var queryKeys, queryDomains = readDomains(fname, false)
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
		if count%1000 == 0 {
			batchTime := time.Since(curQueryingStart)
			println(fmt.Sprintf("In file %v it took %v minutes to query %v domains out of %v (%v%%)", inputFile.Name(), batchTime.Minutes(), count, len(queryKeys), 100.0*float64(count)/float64(len(queryKeys))))
			curQueryingStart = time.Now()
		}
		// query in the domain records:
		querySig := curRecord.Signature
		querySize := curRecord.Size
		var queryKey = curRecord.Key
		var queryKeyString = fmt.Sprintf("%v", queryKey)
		var keyVals = strings.Split(queryKeyString, "||||")
		var id = keyVals[0]
		var version = keyVals[1]
		var attrName = keyVals[2]
		for i := range thresholds {
			done := make(chan struct{})
			defer close(done) // Important!!
			// set the containment startingThreshold
			// get the keys of the candidate domainsToIndex (may contain false positives)
			// through a channel with option to cancel early.
			results := index.Query(querySig, querySize, thresholds[i], done)
			resCount := 0
			for j := range results {
				var resultKeyString = fmt.Sprintf("%v", j)
				var targetKeyVals = strings.Split(resultKeyString, "||||")
				var targetID = targetKeyVals[0]
				var targetVersion = targetKeyVals[1]
				var targetAttrName = targetKeyVals[2]
				println(resultKeyString)
				//TODO: validate results
				resCount++
				if targetID != id || version != targetVersion {
					outFile.WriteString(fmt.Sprintf("%v,%v,%v,%v,%v,%v\n", id, version, attrName, targetID, targetVersion, targetAttrName))
				}
			}
		}
	}
}

func filterByName(infos []os.FileInfo, s string) []os.FileInfo {
	filtered := make([]os.FileInfo, 0, len(infos))
	for _, f := range infos {
		if strings.Contains(f.Name(), s) {
			filtered = append(filtered, f)
		}
	}
	return filtered
}

func writeFileHeader(f *os.File, thresholds []float64) {
	f.WriteString("pageID, tableHID, tableID, column,")
	for i := range thresholds {
		if i == len(thresholds)-1 {
			f.WriteString(fmt.Sprintf("containmentAt_%v\n", thresholds[i]))
		} else {
			f.WriteString(fmt.Sprintf("containmentAt_%v,", thresholds[i]))
		}
	}
}

func addMatchCounts(query []string, candidate []string, thresholds []float64, matchCounts []int) {
	var queryMap = toMap(query)
	var candidateMap = toMap(candidate)
	var containmentCount = 0
	for k := range queryMap {
		if _, ok := candidateMap[k]; ok {
			containmentCount++
		}
	}
	for i := range thresholds {
		if float64(containmentCount)/float64(len(queryMap)) >= thresholds[i] {
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

func readAllDomains(files []os.FileInfo) ([]map[string]bool, []string) {
	var domains []map[string]bool
	var keys []string
	//files = files[1:10]
	var count = 0
	for _, f := range files {
		if count%10 == 0 {
			println(fmt.Sprintf("Read %v of %v files", count, len(files)))
		}
		var fname = os.Args[1] + "/" + f.Name()
		var key, domain = readDomains(fname, false)
		keys = append(keys, key...)
		domains = append(domains, domain...)
		count++
	}
	return domains, keys
}

func readDomains(jsonFile string, keepSingleElem bool) ([]string, []map[string]bool) {
	file, err := os.Open(jsonFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var keys []string
	var domains []map[string]bool

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
		if errJson != nil {
			println("Error while parsing json")
			println(errJson)
			break
		}
		for j := 0; j < len(domainJson.Values); j++ {
			m[domainJson.Values[j]] = true
		}
		if keepSingleElem || len(m) > 1 {
			var domainKey = getDomainKey(domainJson)
			keys = append(keys, domainKey)
			domains = append(domains, m)
		}

	}
	return keys, domains
}

func getDomainKey(domain Domain) string {
	key := fmt.Sprintf("%v||||%v||||%v||||%v", domain.Id, domain.version, domain.AttrName)
	return key
}
