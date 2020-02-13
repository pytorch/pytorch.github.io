#!/bin/sh -e

# set -x #echo on
# inject-google-analytics

echo "Starting Script"
TRACKER_ID="UA-90545585-1"

TRACKER_CODE="
<!-- Google Analytics -->
<script>
(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
})(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

ga('create', '$TRACKER_ID', 'auto');
ga('send', 'pageview');
</script>
<!-- End Google Analytics -->"

echo `dirname $0`
cd `dirname $0`

find . -name '*.html.new' -delete
find . -name '*.html' | egrep -v -- '-frame.html$' | while read -r TARGET; do
  if grep -qi "<frameset" "$TARGET"; then
    continue;
  fi
  if grep -qi "$TRACKER_ID" "$TARGET"; then
    continue;
  fi

  echo "Injecting: $TARGET"

  case `grep -i '</body>' "$TARGET" | wc -l` in
  1)
    cat "$TARGET" | sed "s:\\(<\\/BODY>\\|<\\/body>\\):`echo $TRACKER_CODE | sed 's:;:\\\\;:g' | sed 's/:/\\\\:/g'`\\1:gi" > "$TARGET.new"
    mv -f "$TARGET.new" "$TARGET"
    ;;
  *)
    {
      cat "$TARGET"
      echo "$TRACKER_CODE"
    } > "$TARGET.new"
    mv -f "$TARGET.new" "$TARGET"
    ;;
  esac
done
