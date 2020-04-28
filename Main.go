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
	Version  string   //`json:"Version"''` //TODO: fix this
	AttrName string   //`json:"AttrName"''`
	Values   []string `json:"values"''`
}

func main() {

	programStart := time.Now()
	files, err := ioutil.ReadDir(os.Args[1])

	queryFiles := files
	indexFiles := filterByName(files, "index")
	queryEntireIndex := false
	if len(queryFiles) == 0 {
		queryEntireIndex = true
	}
	//
	indexDomains, keys, keyToDomain := readAllDomains(indexFiles)

	// initializing the domain records to hold the MinHash signatures
	domainRecords := make([]*lshensemble.DomainRecord, len(indexDomains))

	// set the minhash seed
	var seed int64 = 42

	// set the number of hash functions
	numHash := 256

	println("Done Reading files, beginning to processing indexDomains")
	// create the domain records with the signatures
	for i := range indexDomains {
		mh := lshensemble.NewMinhash(seed, numHash)
		for v := range indexDomains[i] {
			mh.Push([]byte(v))
		}
		domainRecords[i] = &lshensemble.DomainRecord{
			Key:       keys[i],
			Size:      len(indexDomains[i]),
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
	if queryEntireIndex {
		var curFileStart = time.Now()
		var curOutFile, _ = os.Create(os.Args[2] + "joinabilityGraph.csv")
		writeFileHeader(curOutFile, thresholds)
		processQueryDomains(indexDomains, seed, numHash, keys, 0, curQueryingStart, "index", index, thresholds, keyToDomain, keyToDomain, curOutFile)
		println(fmt.Sprintf("Total Runtime for querying entire index [h]: %v", time.Since(curFileStart).Hours()))
		curOutFile.Close()
	} else {
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
			processQueryFile(keyToDomain, fname, f, seed, numHash, curQueryingStart, index, thresholds, curOutFile)
			println(fmt.Sprintf("Total Runtime for file %v [h]: %v", f.Name(), time.Since(curFileStart).Hours()))
			curOutFile.Close()
			fileCount++
		}
	}
	println(fmt.Sprintf("Total Runtime for entire Feature Extraction [h]: %v", time.Since(programStart).Hours()))
}

func processQueryFile(indexKeyToDomain map[string]map[string]bool, fname string, inputFile os.FileInfo, seed int64, numHash int, curQueryingStart time.Time, index *lshensemble.LshEnsemble, thresholds []float64, outFile *os.File) {
	var count = 0
	var queryKeyToDomain = make(map[string]map[string]bool)
	var queryKeys, queryDomains = readDomains(fname, queryKeyToDomain, false)
	processQueryDomains(queryDomains, seed, numHash, queryKeys, count, curQueryingStart, inputFile.Name(), index, thresholds, queryKeyToDomain, indexKeyToDomain, outFile)
}

func processQueryDomains(queryDomains []map[string]bool, seed int64, numHash int, queryKeys []string, count int, curQueryingStart time.Time, inputFileName string, index *lshensemble.LshEnsemble, thresholds []float64, queryKeyToDomain map[string]map[string]bool, indexKeyToDomain map[string]map[string]bool, outFile *os.File) {
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
			println(fmt.Sprintf("In file %v it took %v minutes to query %v domains out of %v (%v%%)", inputFileName, batchTime.Minutes(), count, len(queryKeys), 100.0*float64(count)/float64(len(queryKeys))))
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
		done := make(chan struct{})
		defer close(done) // Important!!
		// set the containment startingThreshold
		// get the keys of the candidate domainsToIndex (may contain false positives)
		// through a channel with option to cancel early.
		results := index.Query(querySig, querySize, thresholds[0], done)
		resCount := 0
		for j := range results {
			var resultKeyString = fmt.Sprintf("%v", j)
			var targetKeyVals = strings.Split(resultKeyString, "||||")
			var targetID = targetKeyVals[0]
			var targetVersion = targetKeyVals[1]
			var targetAttrName = targetKeyVals[2]
			//TODO: validate results
			resCount++
			if targetID != id || version != targetVersion {
				queryValues := queryKeyToDomain[queryKeyString]
				resultValues := indexKeyToDomain[resultKeyString]
				results := getTrueResults(queryValues, resultValues, thresholds)
				outFile.WriteString(fmt.Sprintf("%v,%v,%v,%v,%v,%v,", id, version, attrName, targetID, targetVersion, targetAttrName))
				for i := range thresholds {
					if i == len(thresholds)-1 {
						outFile.WriteString(fmt.Sprintf("%v\n", results[i]))
					} else {
						outFile.WriteString(fmt.Sprintf("%v,", results[i]))
					}
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
	f.WriteString("id, Version, attrName, id_fk,version_fk,attrName_fk,")
	for i := range thresholds {
		if i == len(thresholds)-1 {
			f.WriteString(fmt.Sprintf("FK_at_%v\n", thresholds[i]))
		} else {
			f.WriteString(fmt.Sprintf("FK_at_%v,", thresholds[i]))
		}
	}
}

func getTrueResults(query map[string]bool, candidate map[string]bool, thresholds []float64) []bool {
	var results []bool
	var containmentCount = 0
	for k := range query {
		if _, ok := candidate[k]; ok {
			containmentCount++
		}
	}
	for i := range thresholds {
		results = append(results, false)
		if float64(containmentCount)/float64(len(query)) >= thresholds[i] {
			results[i] = true
		}
	}
	return results
}

func toMap(elements []string) map[string]bool {
	elementMap := make(map[string]bool)
	for i := 0; i < len(elements); i += 1 {
		elementMap[elements[i]] = true
	}
	return elementMap
}

func readAllDomains(files []os.FileInfo) ([]map[string]bool, []string, map[string]map[string]bool) {
	var domains []map[string]bool
	var keys []string
	var keyToDomain = make(map[string]map[string]bool)
	//files = files[1:10]
	var count = 0
	for _, f := range files {
		if count%10 == 0 {
			println(fmt.Sprintf("Read %v of %v files", count, len(files)))
		}
		var fname = os.Args[1] + "/" + f.Name()
		var key, domain = readDomains(fname, keyToDomain, false)
		keys = append(keys, key...)
		domains = append(domains, domain...)
		count++
	}
	return domains, keys, keyToDomain
}

func readDomains(jsonFile string, keyToDomain map[string]map[string]bool, keepSingleElem bool) ([]string, []map[string]bool) {
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
			keyToDomain[domainKey] = m
		}

	}
	return keys, domains
}

func getDomainKey(domain Domain) string {
	key := fmt.Sprintf("%v||||%v||||%v||||%v", domain.Id, domain.Version, domain.AttrName)
	return key
}
