package crawler;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import java.io.IOException;

/**
 * Created by Ahmed Badary on 9/12/17.
 * @author Ahmad Badary
 */
public class CollectPages {

    public static String getNewURL(String url) throws IOException {
        Document doc = null;
        try {
            doc = Jsoup.connect(makeURL(url)).get();
//            doc = Jsoup.connect("https://en.wikipedia.org/wiki/Dog").get();
            Elements links = doc.select(".mw-parser-output > p > a");
            String newURL = makeURL(links.first().attr("href"));
            return newURL;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "Failed";
    }

    public static boolean isValidURL(String input) {
        // filters out improperly formatted URLs
        return input.startsWith("http://en.wikipedia.org/wiki/") || input.startsWith("https://en.wikipedia.org/wiki/");
    }

    public static boolean isFirstURL(String link) {
        // filters out most language and pronunciation and non-wiki links
        return (link.contains("wiki") && !link.contains("Greek") && !link.contains("Latin") && !link.contains("wiktionary"));
    }
}