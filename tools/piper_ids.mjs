// Piper (eSpeak) phonemizer — CI aur browser, dono jagah BILKUL ek jaisa.
//
// Kyun: Piper ke VITS models raw huroof nahi lete — woh eSpeak-ng ke IPA
// phonemes lete hain. urduzaban repo mein espeak-ng.js + espeak-ng.wasm
// pehle se maujood hain (18.5 MB), aur us mein Urdu ("ur") ki poori
// support hai — yahan yehi istemal hota hai, taake CI mein banaya gaya
// sample aur browser ki awaz hu-ba-hu ek jaisi ho.
//
// Tareeqa (piper-phonemize/src/phonemize.cpp ke mutabiq):
//   1) matn ko clause (viraam) ke hisaab se toro, viraam alag rakho
//   2) har clause ko eSpeak se IPA banwao  (-q --ipa -v ur)
//   3) NFD (decompose) — "ç" -> "c" + combining cedilla
//   4) har code point -> id  (phoneme_ids.json)
//   5) shuru mein ^ , har phoneme ke baad _ , aakhir mein $
//
// eSpeak har clause ka jawab alag line mein deta hai — is liye saare clause
// ek hi baar mein bheje jaate hain (18.5 MB wasm baar baar load na ho).

const PUNCT = { '۔':'.', '؟':'?', '،':',', '؛':';', '٪':'%',
                '.':'.', '?':'?', '!':'!', ',':',', ';':';', ':':':' };
const CLAUSE_RE = /[۔؟،؛.?!,;:]/;

export function splitClauses(text){
  // -> [{t:'...', p:'.'|''} ...]   p = us clause ke aakhir ka viraam
  const out=[]; let buf='';
  for(const ch of text){
    if(PUNCT[ch] && CLAUSE_RE.test(ch)){
      if(buf.trim()) out.push({t:buf.trim(), p:PUNCT[ch]});
      else if(out.length) out[out.length-1].p = PUNCT[ch];
      buf='';
    } else buf += ch;
  }
  if(buf.trim()) out.push({t:buf.trim(), p:''});
  return out;
}

export function makePhonemizer(ESpeakNG, wasmBinary, idMap){
  let compiled = null;
  async function espeak(text){
    if(!compiled) compiled = await WebAssembly.compile(wasmBinary);
    const out=[];
    await ESpeakNG({
      arguments:['-q','--ipa','-v','ur',text],
      print:s=>out.push(s), printErr:()=>{},
      instantiateWasm:(imports,cb)=>{
        WebAssembly.instantiate(compiled,imports).then(inst=>cb(inst,compiled));
        return {};
      },
    });
    return out;
  }

  // segments: ["satar 1","satar 2",...]  ->  [[id,...],[id,...]]
  return async function toIds(segments){
    const plan=[];                       // har segment ke clauses
    const flat=[];                       // saare clauses ek qatar mein
    for(const s of segments){
      const cs = splitClauses(s);
      plan.push(cs);
      for(const c of cs) flat.push(c.t);
    }
    if(!flat.length) return segments.map(()=>[]);

    // "۔\n" se joro: eSpeak har clause ka jawab apni line mein deta hai
    const lines = await espeak(flat.join('۔\n'));
    if(lines.length !== flat.length)
      throw new Error(`phonemizer: ${flat.length} tukday bheje, ${lines.length} wapas aaye`);

    const id = ch => idMap[ch];
    let k=0;
    return plan.map(cs=>{
      const ids=[];
      const push = ch => { const v=id(ch); if(v!==undefined){ ids.push(v); ids.push(idMap['_']); } };
      ids.push(idMap['^'], idMap['_']);          // bos + pad
      for(const c of cs){
        for(const ch of lines[k++].normalize('NFD')) push(ch);
        if(c.p){ push(c.p); if(c.p===','||c.p===';'||c.p===':') push(' '); }
      }
      ids.push(idMap['$']);                       // eos
      return ids;
    });
  };
}
