(window.webpackJsonp=window.webpackJsonp||[]).push([[2],{BoaQ:function(module,exports){exports.default={"ar":true,"de":true,"es":true,"fr":true,"ja":true,"ko":true,"pseudo":true,"pt":true,"ru":true,"tr":true,"zh":true,"zh-hk":"zh-tw","zh-mo":"zh-tw","zh-tw":true}},HdzH:function(module,a,e){"use strict";e.r(a);var n=e("cDf5"),t=e.n(n),o=/[#!]\{([^}]+?)\}/g;function generateTranslationFunction(a){var e=a||{},n=function f(a,n){var i=e[a]||a;return"object"===t()(n)&&n?i.replace(o,function(a,t){var o=n[t],i=void 0===o?t:o;return e[o]||i}):i};return n.dictionary=e,n.merge=function(){var a=Array.prototype.slice.call(arguments,0),t,o,i;for(t=0;t<a.length;t+=1)if(o=a[t]&&a[t].dictionary)for(i in o)i in e||(e[i]=o[i]);return n},n}a.default=generateTranslationFunction},hS5U:function(module,exports,a){var e=a("BoaQ"),n=e.default?e.default:{},t,o=(0,a("HdzH").default)(n);o.getLocale=function(){return"en"},module.exports=o},"z+Ik":function(module,a,e){"use strict";e.r(a),e.d(a,"languageCodeCSVtoLanguages",function(){return languageCodeCSVtoLanguages}),e.d(a,"languageCodeToName",function(){return languageCodeToName}),e.d(a,"latinizeText",function(){return latinizeText}),e.d(a,"isRightToLeft",function(){return isRightToLeft}),e.d(a,"courseLanguageTags",function(){return u}),e.d(a,"getIetfLanguageTag",function(){return getIetfLanguageTag}),e.d(a,"getLanguageCode",function(){return getLanguageCode}),e.d(a,"getMomentLanguage",function(){return getMomentLanguage}),e.d(a,"getFacebookLocaleString",function(){return getFacebookLocaleString}),e.d(a,"toIetfLanguageTag",function(){return toIetfLanguageTag}),e.d(a,"toLanguageCode",function(){return toLanguageCode}),e.d(a,"getMobileBadgeLanguageCode",function(){return getMobileBadgeLanguageCode}),e.d(a,"supportedLanguageSubdomains",function(){return g});var n=e("VYij"),t=e.n(n),o=e("ML/G"),i=e("hS5U"),r=e.n(i);function languageTagToName(){return{ab:r()("Abkhaz"),aa:r()("Afar"),af:r()("Afrikaans"),ak:r()("Akan"),sq:r()("Albanian"),am:r()("Amharic"),ar:r()("Arabic"),an:r()("Aragonese"),hy:r()("Armenian"),as:r()("Assamese"),av:r()("Avaric"),ae:r()("Avestan"),ay:r()("Aymara"),az:r()("Azerbaijani"),bm:r()("Bambara"),ba:r()("Bashkir"),eu:r()("Basque"),be:r()("Belarusian"),bn:r()("Bengali"),bh:r()("Bihari"),bi:r()("Bislama"),bs:r()("Bosnian"),br:r()("Breton"),bg:r()("Bulgarian"),my:r()("Burmese"),ca:r()("Catalan"),ch:r()("Chamorro"),ce:r()("Chechen"),ny:r()("Chichewa"),zh:r()("Chinese"),"zh-CN":r()("Chinese (Simplified)"),"zh-TW":r()("Chinese (Traditional)"),cv:r()("Chuvash"),kw:r()("Cornish"),co:r()("Corsican"),cr:r()("Cree"),hr:r()("Croatian"),cs:r()("Czech"),da:r()("Danish"),dv:r()("Divehi"),nl:r()("Dutch"),dz:r()("Dzongkha"),en:r()("English"),eo:r()("Esperanto"),et:r()("Estonian"),ee:r()("Ewe"),fo:r()("Faroese"),fj:r()("Fijian"),fi:r()("Finnish"),fr:r()("French"),ff:r()("Fula"),gl:r()("Galician"),ka:r()("Georgian"),de:r()("German"),el:r()("Greek"),gn:r()("Guaraní"),gu:r()("Gujarati"),ht:r()("Haitian"),ha:r()("Hausa"),he:r()("Hebrew"),hz:r()("Herero"),hi:r()("Hindi"),ho:r()("Hiri Motu"),hu:r()("Hungarian"),ia:r()("Interlingua"),id:r()("Indonesian"),ie:r()("Interlingue"),ga:r()("Irish"),ig:r()("Igbo"),ik:r()("Inupiaq"),io:r()("Ido"),is:r()("Icelandic"),it:r()("Italian"),iu:r()("Inuktitut"),ja:r()("Japanese"),jv:r()("Javanese"),kl:r()("Kalaallisut"),kn:r()("Kannada"),kr:r()("Kanuri"),ks:r()("Kashmiri"),kk:r()("Kazakh"),km:r()("Khmer"),ki:r()("Kikuyu"),rw:r()("Kinyarwanda"),ky:r()("Kyrgyz"),kv:r()("Komi"),kg:r()("Kongo"),ko:r()("Korean"),ku:r()("Kurdish"),kj:r()("Kwanyama"),la:r()("Latin"),lb:r()("Luxembourgish"),lg:r()("Ganda"),li:r()("Limburgish"),ln:r()("Lingala"),lo:r()("Lao"),lt:r()("Lithuanian"),lu:r()("Luba-Katanga"),lv:r()("Latvian"),gv:r()("Manx"),mk:r()("Macedonian (FYROM)"),mg:r()("Malagasy"),ms:r()("Malay"),ml:r()("Malayalam"),mt:r()("Maltese"),mi:r()("Māori"),mr:r()("Marathi"),mh:r()("Marshallese"),mn:r()("Mongolian"),na:r()("Nauru"),nv:r()("Navajo"),nb:r()("Norwegian Bokmål"),nd:r()("North Ndebele"),ne:r()("Nepali"),ng:r()("Ndonga"),nn:r()("Norwegian Nynorsk"),no:r()("Norwegian"),ii:r()("Nuosu"),nr:r()("South Ndebele"),oc:r()("Occitan"),oj:r()("Ojibwe"),cu:r()("Old Church Slavonic"),om:r()("Oromo"),or:r()("Oriya"),os:r()("Ossetian"),pa:r()("Panjabi"),pi:r()("Pāli"),fa:r()("Persian"),pl:r()("Polish"),ps:r()("Pashto"),pt:r()("Portuguese (Brazilian)"),"pt-BR":r()("Portuguese (Brazilian)"),"pt-PT":r()("Portuguese (European)"),qu:r()("Quechua"),rm:r()("Romansh"),rn:r()("Kirundi"),ro:r()("Romanian"),ru:r()("Russian"),sa:r()("Sanskrit"),sc:r()("Sardinian"),sd:r()("Sindhi"),se:r()("Northern Sami"),sm:r()("Samoan"),sg:r()("Sango"),sr:r()("Serbian"),gd:r()("Gaelic"),sn:r()("Shona"),si:r()("Sinhala"),sk:r()("Slovak"),sl:r()("Slovene"),so:r()("Somali"),st:r()("Southern Sotho"),es:r()("Spanish"),su:r()("Sundanese"),sw:r()("Swahili"),ss:r()("Swati"),sv:r()("Swedish"),ta:r()("Tamil"),te:r()("Telugu"),tg:r()("Tajik"),th:r()("Thai"),ti:r()("Tigrinya"),bo:r()("Tibetan"),tk:r()("Turkmen"),tl:r()("Tagalog"),tn:r()("Tswana"),to:r()("Tonga"),tr:r()("Turkish"),ts:r()("Tsonga"),tt:r()("Tatar"),tw:r()("Twi"),ty:r()("Tahitian"),ug:r()("Uighur"),uk:r()("Ukrainian"),ur:r()("Urdu"),uz:r()("Uzbek"),ve:r()("Venda"),vi:r()("Vietnamese"),vo:r()("Volapük"),wa:r()("Walloon"),cy:r()("Welsh"),wo:r()("Wolof"),fy:r()("Western Frisian"),xh:r()("Xhosa"),yi:r()("Yiddish"),yo:r()("Yoruba"),za:r()("Zhuang"),zu:r()("Zulu")}}var u=["af","am","ar","az","bg","bn","bs","ca","cs","da","de","el","es","et","eu","en","fa","fi","fr","he","hi","hr","hu","hy","id","it","ja","jv","ka","kk","km","kn","ko","lt","lv","mk","mn","mr","ms","my","ne","nl","no","pl","ps","pt-BR","pt-PT","ro","ru","rw","sk","sl","sq","sr","sv","sw","ta","te","th","tl","tr","uk","ur","uz","vi","yo","zh-CN","zh-TW","zu"],g=["en","de","es","fr","ja","ko","pt","ru","zh","zh-tw"],s={ar:"ar_AR",es:"es_LA"};function toIetfLanguageTag(a){var e;return(a||"").replace(/_/g,"-").split(/[;=,]/)[0].toLowerCase().replace(/(-[a-z]{1,}$)/,function(a){return a.toUpperCase()})}function getIetfLanguageTag(){try{return toIetfLanguageTag(r.a.getLocale())}catch(a){return o.a.push(["user.language.error",{error:a}]),"en-US"}}function toLanguageCode(a){return toIetfLanguageTag(a).split("-")[0]}function getLanguageCode(){return toLanguageCode(getIetfLanguageTag())}function getMomentLanguage(){var a=getIetfLanguageTag().toLowerCase();switch(a){case"zh-tw":return"zh-tw";default:return a.split("-")[0]}}function getFacebookLocaleString(){var a=getIetfLanguageTag().replace("-","_"),e=a.split("_")[0];return s[e]||a}function languageCodeToName(a){var e,n=toIetfLanguageTag(a).split("-"),t=languageTagToName(),o,i;return n.some(function(a,e){var i=n.slice(0,n.length-e).join("-");return!!(o=t[toIetfLanguageTag(i)])})&&null!=o?o:a}function languageCodeCSVtoLanguages(a){var e=(a||"").split(/,\s*/g).filter(Boolean);return t.a.reduce(e,function(a,e){var n=a;return n[e]=languageCodeToName(e),n},{})}function isRightToLeft(a){var e=["ar","he"];return t()(e).contains(toLanguageCode(a))}var l={A:"Á Ă Ắ Ặ Ằ Ẳ Ẵ Ǎ Â Ấ Ậ Ầ Ẩ Ẫ Ä Ǟ Ȧ Ǡ Ạ Ȁ À Ả Ȃ Ā Ą Å Ǻ Ḁ Ⱥ Ã Ɐ ᴀ",AA:"Ꜳ",AE:"Æ Ǽ Ǣ ᴁ",AO:"Ꜵ",AU:"Ꜷ",AV:"Ꜹ Ꜻ",AY:"Ꜽ",B:"Ḃ Ḅ Ɓ Ḇ Ƀ Ƃ ʙ ᴃ",C:"Ć Č Ç Ḉ Ĉ Ċ Ƈ Ȼ Ꜿ ᴄ",D:"Ď Ḑ Ḓ Ḋ Ḍ Ɗ Ḏ ǲ ǅ Đ Ƌ Ꝺ ᴅ",DZ:"Ǳ Ǆ",E:"É Ĕ Ě Ȩ Ḝ Ê Ế Ệ Ề Ể Ễ Ḙ Ë Ė Ẹ Ȅ È Ẻ Ȇ Ē Ḗ Ḕ Ę Ɇ Ẽ Ḛ Ɛ Ǝ ᴇ ⱻ",ET:"Ꝫ",F:"Ḟ Ƒ Ꝼ ꜰ",G:"Ǵ Ğ Ǧ Ģ Ĝ Ġ Ɠ Ḡ Ǥ Ᵹ ɢ ʛ",H:"Ḫ Ȟ Ḩ Ĥ Ⱨ Ḧ Ḣ Ḥ Ħ ʜ",I:"Í Ĭ Ǐ Î Ï Ḯ İ Ị Ȉ Ì Ỉ Ȋ Ī Į Ɨ Ĩ Ḭ ɪ",R:"Ꞃ Ŕ Ř Ŗ Ṙ Ṛ Ṝ Ȑ Ȓ Ṟ Ɍ Ɽ ʁ ʀ ᴙ ᴚ",S:"Ꞅ Ś Ṥ Š Ṧ Ş Ŝ Ș Ṡ Ṣ Ṩ ꜱ",T:"Ꞇ Ť Ţ Ṱ Ț Ⱦ Ṫ Ṭ Ƭ Ṯ Ʈ Ŧ ᴛ",IS:"Ꝭ",J:"Ĵ Ɉ ᴊ",K:"Ḱ Ǩ Ķ Ⱪ Ꝃ Ḳ Ƙ Ḵ Ꝁ Ꝅ ᴋ",L:"Ĺ Ƚ Ľ Ļ Ḽ Ḷ Ḹ Ⱡ Ꝉ Ḻ Ŀ Ɫ ǈ Ł Ꞁ ʟ ᴌ",LJ:"Ǉ",M:"Ḿ Ṁ Ṃ Ɱ Ɯ ᴍ",N:"Ń Ň Ņ Ṋ Ṅ Ṇ Ǹ Ɲ Ṉ Ƞ ǋ Ñ ɴ ᴎ",NJ:"Ǌ",O:"Ó Ŏ Ǒ Ô Ố Ộ Ồ Ổ Ỗ Ö Ȫ Ȯ Ȱ Ọ Ő Ȍ Ò Ỏ Ơ Ớ Ợ Ờ Ở Ỡ Ȏ Ꝋ Ꝍ Ō Ṓ Ṑ Ɵ Ǫ Ǭ Ø Ǿ Õ Ṍ Ṏ Ȭ Ɔ ᴏ ᴐ",OI:"Ƣ",OO:"Ꝏ",OU:"Ȣ ᴕ",P:"Ṕ Ṗ Ꝓ Ƥ Ꝕ Ᵽ Ꝑ ᴘ",Q:"Ꝙ Ꝗ",V:"Ʌ Ꝟ Ṿ Ʋ Ṽ ᴠ",TZ:"Ꜩ",U:"Ú Ŭ Ǔ Û Ṷ Ü Ǘ Ǚ Ǜ Ǖ Ṳ Ụ Ű Ȕ Ù Ủ Ư Ứ Ự Ừ Ử Ữ Ȗ Ū Ṻ Ų Ů Ũ Ṹ Ṵ ᴜ",VY:"Ꝡ",W:"Ẃ Ŵ Ẅ Ẇ Ẉ Ẁ Ⱳ ᴡ",X:"Ẍ Ẋ",Y:"Ý Ŷ Ÿ Ẏ Ỵ Ỳ Ƴ Ỷ Ỿ Ȳ Ɏ Ỹ ʏ",Z:"Ź Ž Ẑ Ⱬ Ż Ẓ Ȥ Ẕ Ƶ ᴢ",IJ:"Ĳ",OE:"Œ ɶ",a:"á ă ắ ặ ằ ẳ ẵ ǎ â ấ ậ ầ ẩ ẫ ä ǟ ȧ ǡ ạ ȁ à ả ȃ ā ą ᶏ ẚ å ǻ ḁ ⱥ ã ɐ ₐ",aa:"ꜳ",ae:"æ ǽ ǣ ᴂ",ao:"ꜵ",au:"ꜷ",av:"ꜹ ꜻ",ay:"ꜽ",b:"ḃ ḅ ɓ ḇ ᵬ ᶀ ƀ ƃ",o:"ɵ ó ŏ ǒ ô ố ộ ồ ổ ỗ ö ȫ ȯ ȱ ọ ő ȍ ò ỏ ơ ớ ợ ờ ở ỡ ȏ ꝋ ꝍ ⱺ ō ṓ ṑ ǫ ǭ ø ǿ õ ṍ ṏ ȭ ɔ ᶗ ᴑ ᴓ ₒ",c:"ć č ç ḉ ĉ ɕ ċ ƈ ȼ ↄ ꜿ",d:"ď ḑ ḓ ȡ ḋ ḍ ɗ ᶑ ḏ ᵭ ᶁ đ ɖ ƌ ꝺ",i:"ı í ĭ ǐ î ï ḯ ị ȉ ì ỉ ȋ ī į ᶖ ɨ ĩ ḭ ᴉ ᵢ",j:"ȷ ɟ ʄ ǰ ĵ ʝ ɉ ⱼ",dz:"ǳ ǆ",e:"é ĕ ě ȩ ḝ ê ế ệ ề ể ễ ḙ ë ė ẹ ȅ è ẻ ȇ ē ḗ ḕ ⱸ ę ᶒ ɇ ẽ ḛ ɛ ᶓ ɘ ǝ ₑ",et:"ꝫ",f:"ḟ ƒ ᵮ ᶂ ꝼ",g:"ǵ ğ ǧ ģ ĝ ġ ɠ ḡ ᶃ ǥ ᵹ ɡ ᵷ",h:"ḫ ȟ ḩ ĥ ⱨ ḧ ḣ ḥ ɦ ẖ ħ ɥ ʮ ʯ",hv:"ƕ",r:"ꞃ ŕ ř ŗ ṙ ṛ ṝ ȑ ɾ ᵳ ȓ ṟ ɼ ᵲ ᶉ ɍ ɽ ɿ ɹ ɻ ɺ ⱹ ᵣ",s:"ꞅ ſ ẜ ẛ ẝ ś ṥ š ṧ ş ŝ ș ṡ ṣ ṩ ʂ ᵴ ᶊ ȿ",t:"ꞇ ť ţ ṱ ț ȶ ẗ ⱦ ṫ ṭ ƭ ṯ ᵵ ƫ ʈ ŧ ʇ",is:"ꝭ",k:"ḱ ǩ ķ ⱪ ꝃ ḳ ƙ ḵ ᶄ ꝁ ꝅ ʞ",l:"ĺ ƚ ɬ ľ ļ ḽ ȴ ḷ ḹ ⱡ ꝉ ḻ ŀ ɫ ᶅ ɭ ł ꞁ",lj:"ǉ",m:"ḿ ṁ ṃ ɱ ᵯ ᶆ ɯ ɰ",n:"ń ň ņ ṋ ȵ ṅ ṇ ǹ ɲ ṉ ƞ ᵰ ᶇ ɳ ñ",nj:"ǌ",oi:"ƣ",oo:"ꝏ",ou:"ȣ",p:"ṕ ṗ ꝓ ƥ ᵱ ᶈ ꝕ ᵽ ꝑ",q:"ꝙ ʠ ɋ ꝗ",u:"ᴝ ú ŭ ǔ û ṷ ü ǘ ǚ ǜ ǖ ṳ ụ ű ȕ ù ủ ư ứ ự ừ ử ữ ȗ ū ṻ ų ᶙ ů ũ ṹ ṵ ᵤ",th:"ᵺ",oe:"ᴔ œ",v:"ʌ ⱴ ꝟ ṿ ʋ ᶌ ⱱ ṽ ᵥ",w:"ʍ ẃ ŵ ẅ ẇ ẉ ẁ ⱳ ẘ",y:"ʎ ý ŷ ÿ ẏ ỵ ỳ ƴ ỷ ỿ ȳ ẙ ɏ ỹ",tz:"ꜩ",ue:"ᵫ",um:"ꝸ",vy:"ꝡ",x:"ẍ ẋ ᶍ ₓ",z:"ź ž ẑ ʑ ⱬ ż ẓ ȥ ẕ ᵶ ᶎ ʐ ƶ ɀ",ff:"ﬀ",ffi:"ﬃ",ffl:"ﬄ",fi:"ﬁ",fl:"ﬂ",ij:"ĳ",st:"ﬆ"},c=t.a.chain(l).map(function(a,e){return t.a.map(a.split(" "),function(a){return[a,e]})}).flatten(!0).object().value();function latinizeText(a){return a.replace(/[^A-Za-z0-9[] ]/g,function(a){return c[a]||a})}function getMobileBadgeLanguageCode(){var a=["en","de","es","fr","ja","ko","pt-br","ru","zh-cn","zh-tw"],e=getIetfLanguageTag().toLowerCase();if(t()(a).contains(e))return e;var n=getLanguageCode();return"pt"===n?"pt-br":"es"===n?"es":"zh"===n?"zh-cn":"en"}a.default={languageCodeCSVtoLanguages:languageCodeCSVtoLanguages,languageCodeToName:languageCodeToName,latinizeText:latinizeText,isRightToLeft:isRightToLeft,courseLanguageTags:u,getIetfLanguageTag:getIetfLanguageTag,getLanguageCode:getLanguageCode,getMomentLanguage:getMomentLanguage,getFacebookLocaleString:getFacebookLocaleString,toIetfLanguageTag:toIetfLanguageTag,toLanguageCode:toLanguageCode,getMobileBadgeLanguageCode:getMobileBadgeLanguageCode,supportedLanguageSubdomains:g}}}]);
//# sourceMappingURL=en.2.bb8c18b185aa4eff80f4.js.map