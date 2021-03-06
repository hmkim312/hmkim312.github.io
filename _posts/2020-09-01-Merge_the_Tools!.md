---
title: Merge the Tools (Python 3)
author: HyunMin Kim
date: 2020-09-01 00:00:00 0000
categories: [Hacker Ranker, Python HR]
tags: []
---

- URL : <https://www.hackerrank.com/challenges/merge-the-tools/problem>{:target="_blank"}

- Consider the following:

    - A string, s, of length n where s = c0c1...cn-1
    - An integer, k, where k is a factor of n.

- We can split s into n/k subsegments where each subsegment, ti, consists of a contiguous block of k characters in s. Then, use each ti to create string ui such that:
    - The characters in ui are a subsequence of the characters in ti.
    - Any repeat occurrence of a character is removed from the string such that each character in ui occurs exactly once. In other words, if the character at some index j in ti occurs at a previous index < j in ti, then do not include the character in string ui.

- Given s and k, print n/k lines where each line i denotes string ui.

- Input Format
    - The first line contains a single string denoting s.
    - The second line contains an integer, k, denoting the length of each subsegment.

- Constraints
    - 1 <= n <= 10^4, where n is the length of s
    - 1 <= k <= n
    - It is guaranteed that n is a multiple of k.
    
- Output Format
    - Print n/k lines where each line i contains string ui.

#### 문제설명
- string s와 배열의 수k가 주어지면 len(s) / k의 갯수만큼 s를 끊어서 보고, 중복되면 제거하는 함수 작성
- 파이썬은 중복 제거로 set을 쓰기는 하지만, 이것은 순서를 보장하지 못함
- 따라서 for문으로 돌리면서 not in 메서드를 활용하여 중복되면 넣지 않는 방법을 사용


```python
s = 'DOWTJAHBJKRXASYLDEQQXLQBFHLZXIKAZHVIJCJUMCUOVSZYYQQXBHYIOKINUUPVBKDNOPJARDQMYQMYIDWLMUTCGDPDYGYBQEOETAGMDWBBONAWSWJGSDIBIZGGFEIVKYBFYHSEGTFUIHTBFCHAHQDQRJWXPGUAHYFFFXJNSRKBAFCIJLIRDLJVWHULOOLVORCWQOJJFVKHAOPKBZDFMMAITWUHMHEBAJXRXGCECOLECDODCTKPFKZZGTIVDPWDYTUXZYDDMQKOTAUYUENZROAZKLUNCQQCLNZNLCSYSCOODKMXRBYKPBLZMGMQYDSMSRZDVRDPUSDZERYVSWFIZRHNZUDZZVLROKWJEABYUGZYYXUQVBYVUIITCOVDJIWYVUJZUMZQTYPTVLJQOKLJSWEWKBBVKXTFTFEOTDGMDXFYKRQZDEKVAQTMSUHCTDJMKNCDSJSXIXVTQNUTREQTWJFOFNSYSBNCQPFKAXHJXECNSYJLEZFALKRQZJRRNETFTBQUYZFXGJLAHWLLXQNIDEBCQSWFWPKPSCLRCSOIBBKKZVXXOLFRRIVHARMWJZOBBBSFUJPXMZIMVFJJQQCIZXRBPGYYFMBGPUMJFBJRBRFYWQJPJLZKIQLZYJXDXBMPXYACBJVEADEZTNQCYPMJHAIGLILJJXYGXAKBKFFKIKLEZBGYSVJAXRIIBVTKTTQQUCTEIZQSLCWKDFKYCTAZLIHODBZYURRQUKUELLWWNVJSCTQENQDAAMOGPPEMHVCDIXHTKFQZCZREUKIJCNLFCCLTRRFAPKFXHMQTUISYJLDGVNPADBFHDSAWJGXYQCXZRQUNYMLKXQQVGFVJICQNUQKFYJGQNDREVLTWXHGXXYSDFPOORGDNZNSXYAQMGHSBVMXSTFRSDJYJAMZRTCGSSBRQBHCHRXLMJIGEOYYTJWCKLDBGGKAYLRQOBVYSSJGDTMJHMHDWGFGTLJZRTZRGSHVTCVNWETZXIIHUSKQARYTCISVBTOKLWHFACUXGOYFYHOSBALBUJWYROUVJIHUGOBJQWIYMGDKPUCQUPTOBPNSFJPORWLAKMJCKRAWXGGPALFUCBLDQYYYJNMCMZCYOLBYEDXEJDTJQAETBRYURWSAMGCYGFYWSBUZFRDOXWYPYEKCVJWMHQOTWQREXSAQTXPYQUPNSNFQTRSQAQEIGSBFIVLGPOYILPJBJYRDNYVGPQYSGCAPUDWEAJMRXKAKBLOMJHPWFLEUBEMJGOACUYIWIVOFHQSJBGVMNNIUAOODSDMATPEPNMOVJEDRUVCXBZJQMTKOJARDFGDBVHSIVGGFKJYGGADJBOZQHLESLWVTCYUZHMJDVPKFAIOGKRPLIQDRBIKPGFIKGEJNSSSPKEVKOJRYCIMKZREAZORFTBPZHAPCTHRFMMPAXIBCRPOSGSUIGMNCPFBXFSZAAQFOFXQEFTIYKXSRRNBABOCSVFPBYQBAJJOOGGUNADOKCGDTVETWVXPSFGTDXXDIITWOBQDBVROXYTRTAMSXMJPRRLWQIZYSUWGVNLZKDPIDIBYKNSHZBXSVIOLSPKKJGSHVFIGIXQLHRMSEEBGIYYDIPOCEAOQJIZGOHOWGEKPVWHCCLIKJINUXEWCGLURVTYJAOGITSXQQHSSSADBLSXKYWOFJIWEBWODLWLGOJWGQPYIRDLEYLPWHDDQOCURYIVLEITURQCJHDTAIHFGSUFZZIPPKLHLUCYAMSXGKZPRCLRLUYTMSYNUHCJTNSEJWFMJXJPJKGANTTANSVZMUMIDQUYGMFRJKDUJMJSZRSMLNOYFJAUFMCKFWJNLOEWYIQHWBZVSTJDGZEOLEISSMDXLMMWARUABMJZNIXHDGLJIRZTVJLNWQNJCZFEQBGTNPTAZQHCYUOILNDGXRWVHJEMILQBMZWBORBOJKSKHGSUUWDTNZQUIUISHBVTASWQJZGVJZFSFXNBVQWIPMFZINTPOPKQJGHUFNPRMUJUUWXPMUYDIFCRUYIJNVBWBISJYIAMELGYKGOYAMCKTGDNEMYTHARKKKTISVXWJFXTLLRNZVYSCBFIPECYGTJQFEBXCYWLFTERGXEFTEXVIEFXKHWQDHJILMINMGARZEIHCPMVTLSBRXBCHZUKGFSTRAHFGHXHOHQSWCPSPJTJGWNPVHZBPUVIUEODNLMDSDVRHMJZVEIEBVTWETATNXCJBSOQDDUXGPQPEZOZGYDHTZFASHTHEXSFPJWUMQUUHKJNMANSYSZURGULOPUSNPAEYYBMQVGZFRNTRALQUMKOTGBJVWDKNDRNBSZRPHRWYGQQGBICOUSJAUUVSXIFDZSERTYGARCYXUQFWYHMUAXUWRSOQATTZNASGYAIPEIOZZTXZCLUEIQABIORIJNJWNBELDMCIXSJWLGXPRTWBKWCSNVDYKMXXQDKTQODNGMLTSKJLEFMQDQISONRYCQYSTILLZRAHDNBWYMJETWUYMERCSKBUAZMVJZHKSHSWVTUVHDZAZTYOXRSRBTLDUZZGBIQTRIPODLJKPIKQDLGDCZUGUIJRHKXKUPGMYXABJMMYWYQAJXDMWZSSHDJRQIBKYHYYHZZQNNQJMGLXDQLBQDWXJFOBQSLOBMMILNYABQJPXVMCLYEBDCBOHRPXLDNNPZXCPYCQONHLIVOWVUZBWCPFUHFFMUUDWSHLSJDGYKSHHIFDEEECJVHDEOKSJHVFBCQTNVZMFTVMDARHGXJPVTUZHHUSOPZRURKJMLVTHSIKUZSDXDSUWPTEWPYMFYDBRPKECIXJAFUVHOAERSAOJVUHLUTSSYTKPDQSNNDOVXLCNLIFFIVRDRYOLUGGUBQKHIEUXHKSHXWUKEBQOYJSPHIBDRHYSZIBJMWGVGZCGVZQCAGQZRKQBTTEMACGBMIKAGTWOUYUPXLTATKBLUTMPNQDPVKTHUFICYGQSGLKGYDGRPJEMCQBPJHHGTANPFXREFHWMUGSSMAMBJQNNJRFUYMATMPIRPZVUGUGDCYVOYJQJAFXLWCFWOFPCUXTLWRGFNMIPNFGNPWZRBYCAAHWQOOUKLNWKGCPTQZLFGRVVPUOSSQSUZRMPFGASVYCCCTXSSIAZBVWQRNLLDDFFWRUDZWWUUYYWUWROGTPKQMAHZNSERZLPSITREPLBPMXLIQAQJPAZDDIEQALJBWZTEUKLLYMAKLMUCOMNDNPHSXNKAYUBVVXBRHMCHAETMQPOGCDKRSTJQGTQFPTCKQFBASFJUKDGCUWJWATNSOWKUSABHWDUMJXMDDYXNBGPVCAUETJXJIJEAKHJILDUWBJBGHATKGKHLLDPGNOQVYWVKDESQKOOLZPTISOSAYCLLFDUUTMPRIMDLRXEDMSQNIJWCXQCXUQLATFUORMIZYLMRLQUZJKMTTKVSBZQVPBXKIUZZGHAEUOWFFSEQDTLYDIREHHCXKZJSVKRBTTIOHGVMYCEBXPAAXTHHCJEMKPEGAXJTQRJZXHMWLSZIKBIHUPOWAUINMOVMNEHFYTFVATTNLTVVUDFRVVNXSYMEOHSCOBHMUMJWFDJTYEQUKVLHRBELBRPPAHTOIDCFPMDXPNSNTIJDEVKXWRIZKZOKIKBSNDXERBDGQVVJGFPMCBJASTZCUPPCZQUOVUTNXYVNWTYCANOEPZEJSFMMUDQWVKMQGHFDGBSCWRGWGUAXTGHNLTBHYRFTDSJKBPPHSILQBROHNOFGWMVJHWQFPYYTSJDTAUDUCQKDKYLXOSGKEBTNAMSPMSIEDNZEICYMTJQDHDCYXKIDMEQOSKDGCOKFDJLNOKAHVSMEXQCUBNXPTPDNZGTDWDIAOWNCGNLDHZHEQKARZXGSMLFNRYQQDBQRXFWDTHGCHPHZBHQAHYUVJZLCZBUCCKWZQSELZMNIEWHFGZGPZCMKCXPBBJGDWCFOWLBXAQGENPLVRTMSWYFAYUDZFJEDNLRJXUHZLPFAGQVZMHSIHXIFTNEAXLEMWXYVSHUFWBFEUCEGJYQSXAYQNESMPWZMWXJQEGVCHCGDEMJPMBHJBHBROUDDSERQBBGHHDJQFSULGFATHKEKRGBHCHLULENPHTWQZIJGCDSKIUDREIEYQHFSOSOCZBRGXPZWZIEBLWMVRRPXBTVTADLRVCTWDMDCDCYDMDGXBUUUMJSPEPJFSUYQYTMCHREKVFNHKWGNQBIEMAURRDYMZWCYRQCYKIIFPYPZWVPMWXRLZLCQQACPYGPPZRPJAZQPZHRVDGIBFBMEOQWEQBVQJLHIEWTGWMYVTPRYXBACEOISFGYVHTORGVBLUXTSJRNEIGDFJDJNRTFYAFWHZKAHHCUBZOTKFJOQRRVBWHQQAYOCDKKEUKMEOGFNWBXEKMUBFPECYVSBTGDWRNBNAPROVZEUADYMRSNXJUBIPTJKBMIUCJKEADUYCYSEEQQXIGUTAXBPQMZUZJODVYHVCBTECLLIDBFLHCHJZIASVBMWLABGZIBDLVHPIUXMWDZFFGPGRPYORKMELNKMXLPKIYARTXFRAEYHNNOEFMSYZGCMTMASZPCIPCBKCGBEMAMZPCFWQZWRFADAODSOUVYMZZWBFAITCWTTYYSQYOIFRLGFRYVNVTZXUYABYIVAEOTEOLVOCDUTQAAJAWXYRWVMUVNUEIUKYQPNDKBHPXAFZBPCZOAQKXCHSSBYAYJZQYOVKRDZPDGQGVSFLUWWRAFMSIKVIVWYVKUFDXGUCNLLKDQWAOURRBDLJQGULEUHOQOUPXOSKZFWFVUFMOYFRCQBSZXFDROUKDOZCETOEYKLWFSKVQPNUHQMGNSKGGGQLWQNALDGKORGVLQSCHFYOXLVMFHVNPLBLDQLPUSBKLJHXBZBIHBWEOTTTCOJRZMEFCSWYMZKXIRWJTZTABPGPLZLPNWHPKNWOFSNRTZRBRPMKPFMSWSJJSWZHSIWCVSTBMGUFFLIYBUKSBWMYPVHJSGQMONRMHKPWSJDZVOYYIISKHEKWZRFTAVIOKBAUNPSFBXGYLGWURQGYUQUVJBPJWZAJACFNUXUVUATIJREAHKYBCSWNTNXRMXCOZJDTGXODAJLJCPJJBJMFDLSYAPRNOUDODHJMEARGJEPOVAXYLMDOXVMXNEMBAQRDZBRDDKMOODCLGAJROMGLKSKZYZCYRUEQXVVAGHOWNTITTRKJGQWSLGRJHTKYNOQMJMORVFOIYYBTPMCXEYPRGGDOANOPBFEMRSEMXUVXSYQJMUHRSYKZFNPHADYDIEQBXUQWONUILMRYHARBZBCGRRPRXPXHUNJRJBPXOLHBYYZHBSIAWLINEZEBOBJIQUBZVRYLEFMCGOLHGWJCJRPNSWRGZAPQWQPRJOEOTQSBEGINCRRLKHYCFPJERAWPSMIECMSXDMZJSJWUBOFNXGSCXBIRDFGVTPBXBTWGGVQAFOVICCXZKPDISMBVTJTOAULCQHKYFAYMPVUTYRVKJAUBOVXHFSVFOGJEQWEVWDHNADJYWGLFGFHVBGCIYAPMIBTYXXTVCCLDFWDCCOHIUOFXUKHUKWJSZETYBMWEPHJWFMYHBGRXWZUTJBPTZYNZCGXGVVMMCVKJJJQMRKJPJDISEYNEYCFDLELIAXUEUEOFPGSGSBWBFGVLGIRHNYKYDVGFUCLQIZVAINGARECWNZJTKDAXBMXGJDNFHZVRATRIGALXGNVTOHPYKRVNEUUPZHVGISAJLRRUTCTARPVIYKGKCEXIYTXYDSGLLGUWAOSTSMTMDPUBBCLDGKLFELFHDNUQUPPUFHPXTLLWCFZDIKJQXUXDHCKLQGDMVSIACXAVKLUMTTQBGBRFXPIFTSSJAVXWQFYSFAPPMLCFFUINVBSUSAZMURXUNVSFCTXHTOXFABMHVUWSYOMQQOCMFCGUXBZZUYJNNGVPHHWEESYCIMUYAZMIBSCAVEBPCKFRSAGZJEFPYFTGUOGWNSEONJOINRARCFLUHRWQYDIWJBDFPJBCEGSRRIBGABXCIKYPEUICYQYJRDOJPPNTXGKOQLWQPUUXGTPKPZMNRNYKQOTFGHADPMUFAQYPMSOVNDFDESSVHRIXHBFNKHTBUPHUHHJWZXRNDWQJQIHXBPXLSEYFLTGFIPZQWKOYKFNPBFYUPHTRWQCRWCWIWEPGURYSBMSLRFCVKBRAKKTHDVAZZYJVCZEYQCTRPLFIRJDBMUBXHUGKRGJTETQIUUHKXADMLKUEUAGGWJDDGJPZPBSVWJEQDNAANFOAQLFKNNQJWUPEDEEVFWSBFWRLLTLYZCBPPIBCVRLTNCYTHCOMAIQGFJTQFERGIUVXCYZZPNUDPUWYWMNZUDHBOAUTGNZPHWOLUQMMDGRUBPTZDGAZJHBZHXTOKUFRRTENJQBOYSJBHEBKKBMVLNUSKPIWKNPDJTQULSLJMWLUAOGKRSHCHBVTRFPDTGGEZYAKQLWDHJZJXHTOCCRLEOHXTYAOFIUEGVRZIPEQYDBVMVLQXEEDULCQJCGOMBVTYOSGDWYBBAYPXLGWQKCKXEAGJHVVKSOIGIQJERMHTMXQXFONRQAOXCWGJTDVLUFRCYCJPQSICPACWQPNHRDGUCMDVRBHLGAQEFZVXTFAJFCHWTXFLANFCBKASLHEUJWAQXWOQBQBJUJHNGOAIBHNCSNUFWABFWCYTAMKBELMYWTNCHOLIVYKPNHVJHYQGAOZACLEGZSGVMUYVKLGFLSXYZUHJSZPTOQVSCBBDWHYIDYDNLLVWESWDMFMEFDZVUXNYAQBWYBGDCMRNXMMBGIFUOTYTXAQTXDRZWUWWXCZZOQNMEZPMJUGXQHSNHIGGOZFKVDGSIHUYAJMGKCTTWZRMGLBPUJXKKDUFICAQLWRLFDSPFLIEMBSVNUMJFKTQPPXXTZQEVHQCMKRSVCWJDQESMRDSBYKSPJPIJHPGOFJBPCVMETVKJCEWVIQYIAQYJHGUPXDFCOGTQDHVWFHGHLEEVUEEWUEFDMCSKHYOVGINKPKIURQDFUJAQNGMJMUNZYFJFFZAOHPYZAIVRZBWVKZOAHAJUWYVUGGZLFBZPRZQRKLKLOJIBKWDRZONVNIQTQPGWTILKJBDTOPEFYNGILLCKZRGMAWHSNOOGYBSHEVDVLHALWIWITYSSPAHRWOKMEASECLMGIPBVYDHXLFFGDZBUAIOYYAKFADJEQXNAOQYPUHMHMRQPSRLSBBRBBBGEGSKYRXBGOZVKGJTSCLJUCWNFYGHBJPFSJSQBPTJFSGRAPNURYDNDCAIAIRBSJJKSBCVSYEASKRTBENUDTIGVKQXTIYNRJZLMDHFBLFUYWNZDCUIXEOSOEPJOQWHZVTOBATCLAYJYNLBQHJPLZIZGZJURHCTFVHGXAKICJUAYHCQONGBNQATPLPJTRCAOJGLLSWNBSNCZRSPFYQSOSNGGCPZWRBMCHAPBWCFQRHRJZGQYYIOQXUYZJXXCAKEJKTLIXQARXRAYZQWXALPZHOYTNYVOICZUXKEXAEOZXOYWGWVIJMHRCHKSFHIPJHLJSRGUVUUSKUQSSLACAJTCQFUYMEPYMDHEUPASJWKVQDNJQPNQYITQNPQBUIBILKMFZOXKLKFDNUODJBWJJRCYGSCCCDKNOYTPNSAAEHDRCSXNVTXFKBFSVHVYLHLBGGQVBQVFYYZCSWPORMVBNAVLISJVAUWIDMDEFAJFAKHTGWJXLEAAGWLOOWJQRHYWUDCBDNGEYPZENKEAQGBWEONVLWLEGMCAPEDVRLBPAAWPMAQCGRYLHNGUMTYSFAUVEXSYITNKVJZHLPLUILHQZPKLILDQNXNUWFSEYHOTRPCEFNYNYHFXWSKHFNXSMKOIPGOPOCKHUMLZBMOBTWBSQLZVBWPNGEYXKMMARWHLKVKLJAMEWPWMBVJCTZRBDPYPCMQTIZEUWQFFQUKPJIDMGNOZMIBRXBIZOYSWAYTWOAEHUQWGYZUEOLGCTJTQKCSYCMXELQDBQHINXEVXGREUCMXVVSOFWGGBSFFDXIFPRPEOWBOCTUYVGXTBSHIONQPHVXLUFSKXHONFQDJJXIGDHZGZGROVHFDFCQZKKJJTYWZQCIZZSHDAHLBPCRNKXSRBKQNUAWPAVQSXZTWTABVHPXZROMDNEURQLGKNDBPASHZRAXMABIKQHJHVXLLDHETUKFHPGYRYHQRHOEKPMUFVFOSCBDHJJADUFNLLLDMTTDCJHMAVIHQPWJTXOAIYCOUJBHXOKJJGMMRWASRLZKCXVWVJYFJDTFMWNJLXUUFJIXFLPYYQIAOFYJPZQACKGOGTATQWQYFAVKLKLLAVMOBMZSLSSPEYGMRGGKEYIKYDWMPJXRFLIGAJYLBSBFRJSIRYSVWAFWDEIUNIMUTUATDAHFTKKKTEUKENHCPPZVTJPIRBDNXFIBGPIBBUNWZJIDYQGOFHLAQAIIELVBRDEZVMCWHPVIADLZTTNAAYARALBGWYKPBOOYDSXMJSUKXHJTBWTDWWWYHXFDVRVZHLXKEWYNOUXMCGFFEAICWFBDEIJCZGBIRAVVXTJNOIZSRGZVJHAFOBLVLWXMCAUTAPRZLCNBKOVBXUZGBZLSCYPNUMBYMVSONLQANDDXTYBQSCWVBKPFIEUFQVDESXVHILHVQKVLKWCCAAABKPITUCAMZDRRBOZJBIFTVAEFYGKYHKBTZLOVNOIMSBFVPGESRLLMNPTNYFNHPQARBPOOFWCZXIUOQBGHORTCJPRHUFOMVQFZFTPNSSMRCJGSMOAAIVERMYAJDQXAGCZOXRBPJPJNAPHMFJPPFVITTICYYZBFDCVBTWSELBRMSZAAIPPPNYJGINHJNIQQMLTGJMKUNEIIDLINCXFPXQYIDHRSPJLDUEJFSUBGYKQDXYSZXZRXRRFXYYPPHASDFCIXYJFYVXDUVXUVZNSQEZNCXDRGFJJMNTMLETLCROWOOQJNFDDJCRLBWEHBQSODNCRSVCUMTSDJLOWQSZAWSNAQSHSICINPMEJHJFWCYZLJNJBHKBGDPGVJPNRRXEJJLUTUAPYAOLLDUMLEQRJFXFQOUJGUPRFALYWNNUPDHAHBNUIDLTKKACYXLGRAZYBLXZAMUQQBQZFFTPKGIWQLYQIJZBMYZPLYOMLICDMUERCXIMDSJVDHNNTMOFNQUYOKKBTOGHLKANJIBOAKLGUZVPNKWACSBSCLUXCCEPPEEYOFPRSAZOBUDRGZRITSBVGXTKZZZOGFPUKENCGMQKJVBPWSAPNBLVAGFBFGSNNHKXNXCVMUFXQGMOBOEQPRLRXSTEALUNUEKJBOEPJKMZSBNVPTLHKWYJQTQQGKDAOQJREOJPAVQUWGQMZBVMATXQPNHVZMWPEFHLVSAXNRSMXIAYJVKJQKCFXLBZXXOEEXPCPPZFITRHBRGNOSYECAKCLLBLKRPOPESGVULENCNRVTGJOGOQHAUULVHXPYNECGMAAYGPCTGYPOJDVZVEZQAMNIJCGXILFXNHVTWXOFXFTHIQGGUGYVUNFGSNDCYKZNRUIPUXURCQYNIHVDPTAJGHPAUUDVECIXZRNTQJMSBNHKUCPJVRUEYMGSIJPPOAMNRBIHNUBOHLADNROLILRIXXDFJSUXSJMLKUUXQYOAJQDWIUJSFABCAEJJYGIRPUCBQZZIXPIGFOCNIOINOJRONCZLIHEXDJZVIAEFROLWCPMMDUBUFSISVKFFTMDXVESDGWIXNVTPKHCODFIIYSCVEIAYUFVRLPUROEOBZKSLRWBXCLFAGIXKSZIMFFFQXZKLFYOGIGRCDTZFEGHMOEXGDHUKNZDMANXFNLLYRFAWYBBEHIRWMQESXZEMYHYZWXGKKSIEZKAXNCDUKWSZOZTMYXAYHZZFYIPJAZNZLNYYPDVCCPDQOWCOVFPEEOKCWZNYBAZMOZLFFIJHXMXOKCEGHTKNKURIWFJZHKLVKYDRGMYGZXWJBARKUEZEATOYYXXFIJDUJGLSSLYTIUFMWWYQCYWDRLBRKBYUMBOVJBNEMNXXJCLGBJYFHXIAKMRUPQRBTFWFJMJ'
k = 1
```


```python
def merge_the_tools(s, k):
    num = int(len(s) / k)
    for i in range(num):
        t = s[i * k : (i + 1) *k]
        c = ''
        for j in t:
            if j not in c:
                c += j
        print(c)
```


```python
merge_the_tools(s, k)
```

    D
    O
    W
    T
    ...
    J



```python
s[i:i+1]
```

    'J'
