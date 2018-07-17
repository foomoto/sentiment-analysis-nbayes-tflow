import scrapy


class AmazonSpider(scrapy.Spider):
    name = 'scrape'
    start_urls = [
        'https://www.amazon.com/Zinus-Mattress-Foundation-structure-assembly/dp/B00NPVCQYO/ref=cm_cr_arp_d_product_top?ie=UTF8',
        'https://www.amazon.com/Zinus-Mattress-Foundation-structure-assembly/dp/B01AS4WAB4/ref=cm_cr_arp_d_product_top?ie=UTF8',
        'https://www.amazon.com/Zinus-Mattress-Foundation-structure-assembly/dp/B01HIR5FYS/ref=cm_cr_arp_d_product_top?ie=UTF8',
        'https://www.amazon.com/Tuft-Needle-Mattress-Certi-PUR-Certified/dp/B00QBZ25SS/ref=pd_sim_196_2?_encoding=UTF8&pd_rd_i=B00QBZ25SS&pd_rd_r=d86908be-880b-11e8-88f0-530297bd69ef&pd_rd_w=uhTvr&pd_rd_wg=xIzEm&pf_rd_i=desktop-dp-sims&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=7967298517161621930&pf_rd_r=RHYQ3R8M5WTC69QCPRSH&pf_rd_s=desktop-dp-sims&pf_rd_t=40701&psc=1&refRID=RHYQ3R8M5WTC69QCPRSH',
        'https://www.amazon.com/Nordcurrent-Ltd-Cooking-Fever/dp/B00TS3HTSG/ref=cm_cr_arp_d_product_top?ie=UTF8']
    text = set()

    def parse(self, response):
        for review in response.css('.a-section .review'):
            review_text = review.css(
                '.a-size-base.a-link-normal.review-title.a-color-base.a-text-bold ::text').extract_first()
            review_rating = (review.css('.a-icon-alt ::text').extract_first()).split(" ")[0]
            if review_text is not None and review_text not in self.text:
                self.text.add(review_text)
                yield {'rating': review_rating, 'text': review_text}
                file = open("ratings.txt", "a")
                file.write(review_rating + ' ' + review_text + '\n')
                file.flush()

        links = response.css('ol > li:nth-child(1) > div > a ::attr(href)').extract() + \
                response.css('a#dp-summary-see-all-reviews ::attr(href)').extract() + \
                response.css('#cm_cr-pagination_bar > ul > li.page-button > a ::attr(href)').extract()
        for next_page in links:
            yield response.follow(next_page, self.parse)
