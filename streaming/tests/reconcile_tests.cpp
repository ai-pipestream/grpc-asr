#include "reconcile.h"
#include <iostream>
#include <tuple>
#include <source_location>
using namespace media;
void check(bool condition, std::source_location at = std::source_location::current()) {
  if (!condition) throw std::runtime_error("reconciliation assertion failed at line " + std::to_string(at.line()));
}
mediapb::SpeechHypothesis window(uint64_t number, uint64_t first, uint64_t fresh,
    std::initializer_list<std::tuple<std::string,int64_t,int64_t>> words) {
  mediapb::SpeechHypothesis h; h.set_window_number(number); h.set_first_sample(first);
  h.set_first_new_sample(fresh); h.set_sample_count(64000);
  auto* s=h.add_segments(); s->set_segment_id(uuid()); s->set_source_language("en");
  for (const auto& [text,start,end] : words) {
    *s->mutable_text() += text; auto* t=s->add_tokens(); t->set_text(text);
    t->set_kind(mediapb::SPEECH_TOKEN_KIND_TEXT); t->set_start_us(start); t->set_end_us(end);
  }
  return h;
}
int main() {
 try {
  std::vector<mediapb::TranscriptSegment> out;
  Reconciler r([&](auto s){out.push_back(std::move(s));});
  r.accept(window(1,0,0,{{" hello",100000,1000000},{" world",3200000,3700000}}),16000);
  check(out.size()==2 && out[0].final() && !out[1].final());
  const auto pending=out[1].segment_id();
  r.accept(window(2,48000,64000,{{" World",3300000,3750000},{" again",4000000,4500000}}),0);
  check(out.size()==3 && out[2].final() && out[2].segment_id()==pending && out[2].revision()==2);
  check(out[2].text()==" World again" && out[2].reconciliation()==mediapb::RECONCILIATION_STATUS_ALIGNED);
  r.finish(); check(out.size()==3);
  bool failed=false; try {r.finish();} catch(const std::logic_error&) {failed=true;} check(failed);
  out.clear(); Reconciler repeats([&](auto s){out.push_back(std::move(s));});
  repeats.accept(window(1,0,0,{{" go",100000,500000},{" go",3300000,3600000}}),16000);
  repeats.accept(window(2,48000,64000,{{" go",3300000,3600000},{" go",5000000,5500000}}),0);
  repeats.finish(); std::string text; for(const auto& s:out) if(s.final()) text+=s.text();
  check(text==" go go go");
  out.clear(); Reconciler uncertain([&](auto s){out.push_back(std::move(s));});
  uncertain.accept(window(1,0,0,{{" left",3300000,3600000}}),16000);
  uncertain.accept(window(2,48000,64000,{{" right",3300000,3600000}}),0);
  uncertain.finish(); text.clear(); for(const auto& s:out) if(s.final()) {text+=s.text(); check(s.reconciliation()==mediapb::RECONCILIATION_STATUS_UNRESOLVED);}
  check(text==" left right");
  out.clear(); Reconciler partial([&](auto s){out.push_back(std::move(s));});
  partial.accept(window(1,0,0,{{" an",3100000,3400000},{" inter",3700000,3990000}}),16000);
  partial.accept(window(2,48000,64000,{{" an",3100000,3400000},{" international",3700000,4500000},{" meeting",4600000,5000000}}),0);
  partial.finish(); text.clear(); for(const auto& s:out) if(s.final()) text+=s.text();
  check(text==" an international meeting");
  out.clear(); Reconciler identity([&](auto s){out.push_back(std::move(s));});
  identity.accept(window(1,0,0,{{" track",3300000,3600000}}),16000);
  auto wrong=window(2,48000,64000,{{" track",3300000,3600000}});
  wrong.set_track_id(1); wrong.mutable_segments(0)->set_track_id(1);
  failed=false; try {identity.accept(wrong,0);} catch(const Failure&) {failed=true;} check(failed);
  check(out.size()==1); // No output attributed to the wrong track.
  Reconciler broken([](auto){throw std::runtime_error("subscriber gone");});
  failed=false; try {broken.accept(window(1,0,0,{{" stop",100000,200000}}),0);} catch(const std::runtime_error&) {failed=true;} check(failed);
  failed=false; try {broken.finish();} catch(const std::logic_error&) {failed=true;} check(failed);
  out.clear(); Reconciler unicode([&](auto s){out.push_back(std::move(s));});
  unicode.accept(window(1,0,0,{{" \xc3",2000000,2500000},{"\xa9",3100000,3400000}}),16000);
  check(out.size()==1 && !out[0].final());
  unicode.finish(); check(out.size()==2 && out[1].final() && out[1].text()==" \xc3\xa9");
  out.clear(); Reconciler ambiguous([&](auto s){out.push_back(std::move(s));});
  ambiguous.accept(window(1,0,0,{{" go",3300000,3600000}}),16000);
  ambiguous.accept(window(2,48000,64000,{{" go",3300000,3600000},{" go",3300000,3600000}}),0);
  ambiguous.finish(); text.clear(); for(const auto& s:out) if(s.final()) {text+=s.text(); check(s.reconciliation()==mediapb::RECONCILIATION_STATUS_UNRESOLVED);}
  check(text==" go go go");
  out.clear(); Reconciler timing([&](auto s){out.push_back(std::move(s));});
  timing.accept(window(1,0,0,{{" old",3300000,4100000}}),16000); // Out-of-window native timing.
  timing.accept(window(2,48000,64000,{{" old",3300000,3900000},{" tail",6300000,6700000}}),16000);
  timing.finish();
  check(out.back().final() && out.back().text()==" tail" && out.back().reconciliation()==mediapb::RECONCILIATION_STATUS_UNRESOLVED);
  // Native vocabulary bytes are retained while rendered text drives alignment.
  out.clear(); Reconciler pieces([&](auto s){out.push_back(std::move(s));});
  auto first=window(1,0,0,{{" hello",100000,1000000},{" world",3200000,3700000}});
  auto* token=first.mutable_segments(0)->mutable_tokens(1);
  token->set_text("▁world"); token->set_rendered_text(" world"); token->set_token_id(42);
  pieces.accept(first,16000);
  auto second=window(2,48000,64000,{{"World",3300000,3750000},{" again",4000000,4500000}});
  token=second.mutable_segments(0)->mutable_tokens(0);
  token->set_text("▁World"); token->set_rendered_text("World"); token->set_token_id(43);
  pieces.accept(second,0); pieces.finish();
  check(out.size()==3 && out.back().final() && out.back().text()=="World again");
  check(out.back().reconciliation()==mediapb::RECONCILIATION_STATUS_ALIGNED);
  check(out.back().tokens(0).text()=="▁World" && out.back().tokens(0).rendered_text()=="World" &&
        out.back().tokens(0).token_id()==43);
  // Presence matters: an explicitly empty rendering must not expose vocabulary markers.
  out.clear(); Reconciler empty_piece([&](auto s){out.push_back(std::move(s));});
  auto empty=window(1,0,0,{{"",100000,200000},{"word",200000,300000}});
  token=empty.mutable_segments(0)->mutable_tokens(0);
  token->set_text("▁"); token->set_rendered_text("");
  empty_piece.accept(empty,0); empty_piece.finish();
  check(out.size()==1 && out[0].text()=="word" && out[0].tokens(0).text()=="▁");
  // Native byte tokens can render partial UTF-8; only complete text may be finalized.
  out.clear(); Reconciler rendered_unicode([&](auto s){out.push_back(std::move(s));});
  auto split=window(1,0,0,{{" \xc3",2000000,2500000},{"\xa9",3100000,3400000}});
  token=split.mutable_segments(0)->mutable_tokens(0);
  token->set_rendered_text(token->text()); token->set_text("<0xC3>");
  token=split.mutable_segments(0)->mutable_tokens(1);
  token->set_rendered_text(token->text()); token->set_text("<0xA9>");
  rendered_unicode.accept(split,16000);
  check(out.size()==1 && !out[0].final());
  rendered_unicode.finish();
  check(out.size()==2 && out[1].final() && out[1].text()==" \xc3\xa9" &&
        out[1].tokens(0).text()=="<0xC3>" && out[1].tokens(1).text()=="<0xA9>");
  // Both representations count against the bound, including native bytes that render empty.
  Reconciler huge([](auto){});
  auto oversized=window(1,0,0,{{"",100000,200000}});
  token=oversized.mutable_segments(0)->mutable_tokens(0);
  token->set_text(std::string(128*1024+1,'x')); token->set_rendered_text("");
  failed=false; try {huge.accept(oversized,0);} catch(const Failure& e) {
    failed=e.code==grpc::StatusCode::RESOURCE_EXHAUSTED;
  } check(failed);
  Reconciler mismatch([](auto){});
  auto bad=window(1,0,0,{{"word",100000,200000}});
  bad.mutable_segments(0)->mutable_tokens(0)->set_rendered_text("different");
  failed=false; try {mismatch.accept(bad,0);} catch(const Failure& e) {
    failed=e.code==grpc::StatusCode::DATA_LOSS;
  } check(failed);
  // Overlapping audio can yield speech only beyond the preceding window.
  out.clear(); Reconciler disjoint([&](auto s){out.push_back(std::move(s));});
  disjoint.accept(window(1,0,0,{{" earlier",3300000,3600000}}),16000);
  disjoint.accept(window(2,48000,64000,{{" later",4500000,4900000}}),0);
  disjoint.finish(); text.clear();
  for(const auto& s:out) if(s.final()) {
    check(s.reconciliation()==mediapb::RECONCILIATION_STATUS_DIRECT); text+=s.text();
  }
  check(text==" earlier later");
  // An empty hypothesis has no competing text; retain the preceding tail.
  out.clear(); Reconciler no_text([&](auto s){out.push_back(std::move(s));});
  no_text.accept(window(1,0,0,{{" retained",3300000,3600000}}),16000);
  const auto retained_id=out.back().segment_id();
  no_text.accept(window(2,48000,64000,{}),0); no_text.finish();
  check(out.size()==2 && out.back().final() && out.back().text()==" retained" &&
        out.back().segment_id()==retained_id && out.back().revision()==2 &&
        out.back().reconciliation()==mediapb::RECONCILIATION_STATUS_DIRECT);
  // SentencePiece strips the first word's space in each independent window.
  out.clear(); Reconciler word_starts([&](auto s){out.push_back(std::move(s));});
  auto start_word=window(1,0,0,{{"First",100000,500000}});
  token=start_word.mutable_segments(0)->mutable_tokens(0);
  token->set_text("▁First"); token->set_rendered_text("First");
  token->mutable_parakeet()->set_is_word_start(true);
  word_starts.accept(start_word,0);
  auto next_word=window(2,64000,64000,{{"Next",4500000,4900000}});
  token=next_word.mutable_segments(0)->mutable_tokens(0);
  token->set_text("▁Next"); token->set_rendered_text("Next");
  token->mutable_parakeet()->set_is_word_start(true);
  const auto raw_next=next_word.SerializeAsString();
  word_starts.accept(next_word,0); word_starts.finish();
  check(out.size()==2 && out[0].text()+out[1].text()=="First Next" &&
        out[1].tokens(0).text()=="▁Next" && out[1].tokens(0).rendered_text()==" Next" &&
        next_word.SerializeAsString()==raw_next);
  // A stable cutoff must not finalize the first piece of a word whose tail overlaps.
  out.clear(); Reconciler split_word([&](auto s){out.push_back(std::move(s));});
  auto before_cut=window(1,0,0,{{"your",1000000,1500000},{" co",2800000,2960000},
      {"untry",2960000,3200000},{" can",3200000,3450000}});
  for (int i=0; i<before_cut.segments(0).tokens_size(); ++i)
    before_cut.mutable_segments(0)->mutable_tokens(i)->mutable_parakeet()->set_is_word_start(i!=2);
  split_word.accept(before_cut,16000);
  check(out.size()==2 && out[0].final() && out[0].text()=="your" &&
        !out[1].final() && out[1].text()==" country can");
  auto after_cut=window(2,48000,64000,{{"co",3000000,3100000},{"untry",3100000,3250000},
      {" can",3250000,3500000},{" do",4000000,4200000}});
  for (int i=0; i<after_cut.segments(0).tokens_size(); ++i)
    after_cut.mutable_segments(0)->mutable_tokens(i)->mutable_parakeet()->set_is_word_start(i!=1);
  split_word.accept(after_cut,0); split_word.finish(); text.clear();
  for (const auto& s:out) if(s.final()) {
    check(s.reconciliation()!=mediapb::RECONCILIATION_STATUS_UNRESOLVED); text+=s.text();
  }
  check(text=="your country can do");
  std::cout<<"reconciliation cases passed\n";
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
