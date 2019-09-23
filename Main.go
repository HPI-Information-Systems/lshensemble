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
	"strconv"
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


	//create int aliases for out Files to reduce memory usage of domainKeys
	var idToBucketName = make(map[int]string)
	var bucketNameToid = make(map[string]int)
	fileID :=0
	for _, file := range files {
		idToBucketName[fileID] = file.Name()
		bucketNameToid[file.Name()] = fileID
		fileID++
	}
	for _, file := range trainFiles {
		idToBucketName[fileID] = file.Name()
		bucketNameToid[file.Name()] = fileID
		fileID++
	}
	//result file writing new:
	var outFiles = make(map[int]*os.File)
	for bID, name := range idToBucketName {
		curFile,err := os.Create(os.Args[3] + name + "_features.csv")
		if err != nil {
			fmt.Println("could not create output file")
			fmt.Println(err)
			return
		}
		outFiles[bID] =curFile
	}

	//
	domainsToIndex, keys, _ := readAllDomains(idToBucketName,bucketNameToid,"index")

	//create by hand:
	//domainsToIndex, keys := buildSmallTestDomains()

	// FIRST STEP

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

	var keyToDomainValues = make(map[string]map[string]bool)
	for i:= range domainsToIndex {
		var curKey = keys[i]
		keyToDomainValues[curKey] = domainsToIndex[i]
	}

	// Create index using equi-depth partitioning
	// You can also use BootstrapLshEnsemblePlusEquiDepth for better accuracy
	//index, err := lshensemble.BootstrapLshEnsembleEquiDepth(numPart, numHash, maxK,
	//	len(domainRecords), lshensemble.Recs2Chan(domainRecords))
	//if err != nil {
	//	panic(err)
	//}

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
	serializeIndex(index)
	println("Done Indexing")
	indexingTime := time.Since(programStart)
	println(fmt.Sprintf("Build Index in %v hours",indexingTime.Hours()))
	curQueryingStart := time.Now()
	//THIRD STEP
	//Chose Query:
	//var queryDomain = Domain{json.Number(27439),"Suriname National Army",json.Number(586117314),"333936538-0","Origin",1,[] string{}}
	var count =0
	//TODO: result file writing:
	if err != nil {
		fmt.Println(err)
		return
	}

	//file pointer for train files:

	var thresholds [] float64
	thresholds = append(thresholds,0.5)
	thresholds = append(thresholds,0.6)
	thresholds = append(thresholds,0.7)
	thresholds = append(thresholds,0.8)
	thresholds = append(thresholds,0.9)
	thresholds = append(thresholds,1.0)
	for _,f := range outFiles{
		writeFileHeader(f, thresholds)
	}
	//sinlgeOutFile,err := os.Create(os.Args[3] + "features.csv")
	//writeFileHeader(sinlgeOutFile,thresholds)
	//fmt.Println("indexQueryTime,resultConfirmationTime,SerializationTime")
	queryTimes,err := os.Create("queryTimes.csv")
	queryTimes.WriteString("batchNumber,Time [s]\n")
	for i := range domainsToIndex {
		//startIteration := time.Now()
		count++
		if (count%1000 == 0) {
			batchTime := time.Since(curQueryingStart)
			println(fmt.Sprintf("Took %v minutes to query %v out of %v (%v%%)",batchTime.Minutes(), count, len(keys), 100.0*float64(count)/float64(len(keys))))
			curQueryingStart = time.Now()
		}
		var queryIndex = i
		// query in the domain records:
		querySig := domainRecords[queryIndex].Signature
		querySize := domainRecords[queryIndex].Size
		var queryKey = domainRecords[queryIndex].Key
		var queryKeyString = fmt.Sprintf("%v", queryKey)
		var queryValues = getKeys(keyToDomainValues[queryKeyString])
		// set the containment startingThreshold
		startingThreshold := 0.5
		// get the keys of the candidate domainsToIndex (may contain false positives)
		// through a channel with option to cancel early.
		done := make(chan struct{})
		defer close(done) // Important!!
		results := index.Query(querySig, querySize, startingThreshold, done)
		var matchCounts [] int
		for i := range thresholds {
			matchCounts = append(matchCounts, 0*i)
		}
		//indexQueryTime := time.Since(startIteration)
		//startResultConfirmation := time.Now()
		resCount :=0
		for key := range results {
			// ...
			// You may want to include a post-processing step here to remove
			// false positive domainsToIndex using the actual domain values.
			// ...
			// You can call break here to stop processing results.
			var resultKeyAsString = fmt.Sprintf("%v", key)
			curResultValues := getKeys(keyToDomainValues[resultKeyAsString])
			addMatchCounts(queryValues, curResultValues, thresholds, matchCounts)
			resCount++
		}
		//resultConfirmationTime := time.Since(startResultConfirmation)
		//startSerialization := time.Now()
		//fmt.Println("Time since start of iteration: %v", time.Since(startIteration))
		var keyVals = strings.Split(queryKeyString, "||||")
		var bucketID,_ =  strconv.Atoi(keyVals[0])
		var pageID = keyVals[1]
		var tableHID = keyVals[2]
		var tableID = keyVals[3]
		var column = keyVals[4]
		//result writing new:
		var f = outFiles[bucketID]
		f.WriteString(fmt.Sprintf("%v,%v,%v,%v,", pageID, tableHID, tableID, column))
		for i := range thresholds {
			if (i == len(thresholds)-1) {
				f.WriteString(fmt.Sprintf("%v\n", matchCounts[i]))
			} else {
				f.WriteString(fmt.Sprintf("%v,", matchCounts[i]))
			}
		}
		//result writing old:
		/*sinlgeOutFile.WriteString(fmt.Sprintf("%v,%v,%v,%v,%v,",idToBucketName[bucketID], pageID, tableHID, tableID, column))
		for i := range thresholds {
			if (i == len(thresholds)-1) {
				sinlgeOutFile.WriteString(fmt.Sprintf("%v\n", matchCounts[i]))
			} else {
				sinlgeOutFile.WriteString(fmt.Sprintf("%v,", matchCounts[i]))
			}
		}*/
		//SerializationTime := time.Since(startSerialization)
		//fmt.Printf("%v,%v,%v,%v\n", resCount,indexQueryTime.Seconds(),resultConfirmationTime.Seconds(),SerializationTime.Seconds())

	}
	/*for curQueryKey:= range myTableKeys {
		count++
		if(count % 1000 == 0){
			println(fmt.Sprintf("Queried %v out of %v",count,len(myTableKeys)))
		}
		var queryIndex = Find(keys,curQueryKey)
		// query in the domain records:
		querySig := domainRecords[queryIndex].Signature
		querySize := domainRecords[queryIndex].Size
		var queryKey = domainRecords[queryIndex].Key
		var queryKeyString = fmt.Sprintf("%v",queryKey)
		var queryValues = getKeys(keyToDomainValues[queryKeyString])
		// set the containment startingThreshold
		startingThreshold := 0.5
		// get the keys of the candidate domainsToIndex (may contain false positives)
		// through a channel with option to cancel early.
		done := make(chan struct{})
		defer close(done) // Important!!
		results := index.Query(querySig, querySize, startingThreshold, done)
		var matchCounts [] int
		for i := range thresholds {
			matchCounts = append(matchCounts,0*i)
		}
		for key := range results {
			// ...
			// You may want to include a post-processing step here to remove
			// false positive domainsToIndex using the actual domain values.
			// ...
			// You can call break here to stop processing results.
			var resultKeyAsString = fmt.Sprintf("%v", key)
			curResultValues := getKeys(keyToDomainValues[resultKeyAsString])
			var isNotFromQueries = !contains(myTableKeys,resultKeyAsString)
			if(isNotFromQueries){
				addMatchCounts(queryValues,curResultValues,thresholds,matchCounts)
			}
		}
		var keyVals = strings.Split(curQueryKey,"||||")
		var sequence = keyVals[0]
		var table = keyVals[2]
		var column = keyVals[4]
		f.WriteString(fmt.Sprintf("%v,%v,%v,",sequence,table,column))
		for i:= range thresholds{
			if(i==len(thresholds)-1){
				f.WriteString(fmt.Sprintf("%v\n",matchCounts[i]))
			} else{
				f.WriteString(fmt.Sprintf("%v,",matchCounts[i]))
			}
		}
	}*/
	for _,f := range outFiles {
		err = f.Close()
		if err != nil {
			fmt.Println(err)
			return
		}
	}
	queryTimes.Close()
}

func serializeIndex(ensemble *lshensemble.LshEnsemble) {
	indexOut,err := os.Create("index.json")
	json,err := json.Marshal(ensemble)
	if err != nil {
		fmt.Printf("Error: %s", err)
		return;
	}
	indexOut.WriteString(string(json))
	indexOut.Close()
}

func writeFileHeader(f *os.File, thresholds []float64) {
	f.WriteString("Sequence,Table,Column,")
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

func contains(m map[string]bool, s string) bool {
	if _, ok := m[s]; ok { return true} else {return false}
}

func toMap(elements []string) map[string]bool {
	elementMap := make(map[string]bool)
	for i := 0; i < len(elements); i += 1 {
		elementMap[elements[i]] = true
	}
	return elementMap
}



func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys;
}

// Find returns the smallest index i at which x == a[i],
// or len(a) if there is no such index.
func Find(a []string, x string) int {
	for i, n := range a {
		if x == n {
			return i
		}
	}
	return len(a)
}

func readLines(s string) interface{} {
	file, err := os.Open(s)
		if err != nil {
			log.Fatal(err)
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		var values [] string
		for scanner.Scan() {
		values = append(values,scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return values
}

var idToBucketName = make(map[int]string)
var bucketNameToid = make(map[string]int)

func readAllDomains(idToBucketName map[int]string,bucketNameToid map[string]int,inputType string) ([]map[string]bool, []string, map[string]bool) {
	var domains [] map[string]bool
	var keys [] string
	files, err := ioutil.ReadDir(os.Args[1])
	trainFiles, err2 := ioutil.ReadDir(os.Args[2])
	if err != nil || err2 != nil{
		log.Fatal(err)
	}
	//files = files[1:10]
	var count = 0
	for _, f := range files {
		if(strings.Contains(f.Name(),inputType)) {
			if (count%10 == 0) {
				println(fmt.Sprintf("Read %v of %v files", count, len(files)))
			}
			var fname= os.Args[1] + "/" + f.Name()
			var key, domain= readDomains(fname, bucketNameToid[f.Name()], false)
			keys = append(keys, key...)
			domains = append(domains, domain...)
			count++
		}
	}
	var myTableKeys = make(map[string]bool)
	for _, f := range trainFiles {
		println(fmt.Sprintf("Reading %v", f))
		var fname = os.Args[2] + "/" + f.Name()
		var key,domain = readDomains(fname,bucketNameToid[f.Name()],true)
		for i := range key{
			myTableKeys[key[i]] = true
		}
		keys = append(keys, key...)
		domains = append(domains, domain...)
	}
	return domains,keys, myTableKeys
}

func readDomains(jsonFile string,bucketID int, keepSingleElem bool) ([]string,[]map[string]bool) {
	file, err := os.Open(jsonFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	//
	//scanner := bufio.NewScanner(file)
	//var m = make(map[string]bool)
	//for scanner.Scan() {
	//	m[scanner.Text()] = true
	//}
	//if err := scanner.Err(); err != nil {
	//	log.Fatal(err)
	//}
	//return m
	// read our opened xmlFile as a byte array.
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
		//filter out some sonsense:
		//_, ok := candidateMap[k]; ok
		//var containsEmptyString = false
		//if _, ok := m[""]; ok { containsEmptyString = true}
		if(keepSingleElem || len(m)>1) {
			var domainKey = getDomainKey(bucketID,domainJson)
			keys = append(keys,domainKey)
			domains = append(domains, m)
		}

	}


	//OLD CODE
	/*byteValue, _ := ioutil.ReadAll(file)


	err = json.Unmarshal(byteValue, &domainsJson)
	for i := 0; i < len(domainsJson.Domains); i++ {
		var domainJson = domainsJson.Domains[i]
		var m = make(map[string]bool)
		for j := 0; j < len(domainJson.Values); j++ {
			m[domainJson.Values[j]] = true
		}
		//filter out some sonsense:
		//_, ok := candidateMap[k]; ok
		//var containsEmptyString = false
		//if _, ok := m[""]; ok { containsEmptyString = true}
		if(keepSingleElem || len(m)>1) {
			var domainKey = getDomainKey(domainJson)
			keys = append(keys,domainKey)
			domains = append(domains, m)
		}
	}*/
	return keys,domains
}

func getDomainKey(bID int,domain Domain) string {
	return fmt.Sprintf("%v||||%v||||%v||||%v||||%v",bID, domain.PageID, domain.TableHID, domain.TID, domain.ColID)
}

